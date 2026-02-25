WITH params AS (
    SELECT
        -- for better performance, query by day, or with even shorter windows
        -- (transfers table is enormous, especially at peak activity periods)
        TIMESTAMP '2024-10-01 00:00:00 UTC' AS start_ts,
        TIMESTAMP '2024-10-02 00:00:00 UTC' AS end_ts,
        CAST('So11111111111111111111111111111111111111112' AS VARCHAR) AS wsol,
        INTERVAL '1' HOUR AS horizon,
        INTERVAL '10' MINUTE AS lookback
),

token_universe_slice AS (
    SELECT
        token_mint,
        pool_id,
        token_vault,
        quote_vault,
        CAST(first_trade_time AS TIMESTAMP) AS first_trade_time
    -- uploaded dataset with tokens for which features can be calculated
    FROM dune..., params
    WHERE CAST(first_trade_time AS TIMESTAMP) >= start_ts
        and CAST(first_trade_time AS TIMESTAMP) < end_ts
),

flattened_adresses AS (
    SELECT token_mint, first_trade_time, 'base_vault' AS vault_role, token_vault AS vault_acct, token_mint AS mint
    FROM token_universe_slice
    WHERE token_vault IS NOT NULL and token_vault != ''

    UNION ALL

    SELECT token_mint, first_trade_time, 'quote_vault' AS vault_role, quote_vault AS vault_acct, wsol AS mint
    FROM token_universe_slice, params
    WHERE quote_vault IS NOT NULL and quote_vault != ''
),

vault_transfers AS (
    SELECT
        f.token_mint, f.first_trade_time, f.vault_role,
        t.block_slot, t.block_time, t.tx_id, t.tx_index,

        CASE
            WHEN t.to_token_account = f.vault_acct THEN CAST(t.amount AS DECIMAL(38,0))
            WHEN t.from_token_account = f.vault_acct THEN -CAST(t.amount AS DECIMAL(38,0))
            ELSE CAST(0 AS DECIMAL(38,0))
        END AS signed_amount_raw

    FROM tokens_solana.transfers t
    JOIN flattened_adresses f
    ON
        t.token_mint_address = f.mint
        and (t.from_token_account = f.vault_acct OR t.to_token_account = f.vault_acct)
    CROSS JOIN params
    WHERE
        -- first level quick filter by date
        t.block_date >= DATE(start_ts - lookback)
        and t.block_date <= DATE(end_ts + horizon)
        -- token level time filter
        and t.block_time >= f.first_trade_time - lookback
        and t.block_time < f.first_trade_time + horizon
        -- removing strange records
        and t.from_token_account != t.to_token_account
        and lower(t.action) LIKE 'transfer%'
),

tx_agg AS (
    SELECT
        token_mint, tx_id,
        arbitrary(first_trade_time) AS first_trade_time,
        MIN(block_slot) AS block_slot,
        MIN(block_time) AS block_time,

        MIN(COALESCE(tx_index, 0)) AS min_tx_index,
        MIN(COALESCE(outer_instruction_index, 0)) AS min_outer_instruction_index,
        MIN(COALESCE(inner_instruction_index, 0)) AS min_inner_instruction_index,

        SUM(CASE WHEN vault_role = 'base_vault' THEN signed_amount_raw ELSE 0 END) AS delta_base_vault_raw,
        SUM(CASE WHEN vault_role = 'quote_vault' THEN signed_amount_raw ELSE 0 END) AS delta_quote_vault_raw
    FROM vault_transfers
    GROUP BY token_mint, tx_id
),

ordered AS (
    SELECT
        token_mint, delta_base_vault_raw, delta_quote_vault_raw,
        row_number() OVER (
            PARTITION BY token_mint
            -- Solana transactions order 
            ORDER BY block_slot, min_tx_index, min_outer_instruction_index, min_inner_instruction_index, tx_id
        ) AS seq,

        -- simplified timestamp: seconds since first trade
        CAST(to_unixtime(block_time) - to_unixtime(first_trade_time) AS INTEGER) AS t_rel_s
    FROM tx_agg
),

reserves AS (
    SELECT *,
        LAG(t_rel_s) OVER (PARTITION BY token_mint ORDER BY seq) AS prev_t_rel_s,
        SUM(delta_base_vault_raw) OVER w AS base_liquidity_raw,
        SUM(delta_quote_vault_raw) OVER w AS quote_liquidity_raw
    FROM ordered
    WINDOW w AS (PARTITION BY token_mint ORDER BY seq ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
),

priced AS (
    SELECT *,
        CAST(quote_liquidity_raw AS DOUBLE) / NULLIF(CAST(base_liquidity_raw AS DOUBLE), 0.0) AS midquote
    FROM reserves
),

peaks_and_ends AS (
    SELECT *,
        MAX(midquote) OVER w as max_price,
        MAX_BY(seq, midquote) OVER w as max_price_idx,
        MAX_BY(midquote, seq) OVER w as last_price,
        MAX(quote_liquidity_raw) OVER w as max_tvl,
        MAX_BY(seq, quote_liquidity_raw) OVER w as max_tvl_idx,
        MAX_BY(quote_liquidity_raw, seq) OVER w as last_tvl,
        MAX_BY(t_rel_s, seq) OVER w as last_t_rel_s
    FROM priced
    WHERE t_rel_s >= 0
    WINDOW w AS (PARTITION BY token_mint)
)

SELECT
    token_mint,
    arbitrary(max_price) as max_price, arbitrary(last_price) as last_price, arbitrary(max_price_idx) as max_price_idx,
    arbitrary(max_tvl) as max_tvl, arbitrary(last_tvl) as last_tvl, arbitrary(max_tvl_idx) as max_tvl_idx,
    MIN(midquote) as min_price,
    MIN(quote_liquidity_raw) as min_tvl,

    -- aggregate on data stricly after the peak (NULL if peak was at the end of 1 hour)
    MIN(CASE WHEN seq > max_price_idx THEN midquote ELSE NULL END) as posterior_min_price,
    MIN(CASE WHEN seq > max_tvl_idx THEN quote_liquidity_raw ELSE NULL END) as posterior_min_tvl,

    -- inactivity either between transfers, or from last transfer to end of 1 hour
    GREATEST(MAX(COALESCE(t_rel_s - prev_t_rel_s, 0)), 60 * 60 - MAX(t_rel_s)) as longest_inactivity
FROM peaks_and_ends
GROUP BY token_mint
-- no activity filter – already built into our uploaded dataset
