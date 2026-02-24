WITH params AS (
    SELECT
        -- for better performance, query by day, or with even shorter windows
        -- (transfers table is enormous, especially at peak activity periods)
        TIMESTAMP '2024-10-01 00:00:00 UTC' AS start_ts,
        TIMESTAMP '2024-10-02 00:00:00 UTC' AS end_ts,
        CAST('So11111111111111111111111111111111111111112' AS VARCHAR) AS wsol_mint,
        INTERVAL '5' MINUTE AS horizon,
        -- have to look for transfers before the first trade to find pool liquidity creation
        INTERVAL '10' MINUTE AS lookback
),

token_universe_slice AS (
    SELECT
        token_mint,
        pool_id,
        token_vault,
        quote_vault,
        CAST(first_trade_time AS TIMESTAMP) AS first_trade_time
    FROM dune.niiiik.dataset_temp, params -- uploaded dataset with enriched pools
    WHERE 
        CAST(first_trade_time AS TIMESTAMP) >= start_ts
        and CAST(first_trade_time AS TIMESTAMP) < end_ts
),

-- to save on output size, we convert the token_mint into
-- a unique alphabetic id within the tokens subset
alphabetic AS (
    SELECT
        row_number() OVER (ORDER BY token_mint) AS token_id,
        token_mint,
        token_vault,
        quote_vault,
        first_trade_time
    FROM token_universe_slice
),

flattened_adresses AS (
    SELECT token_id, first_trade_time, 'base_vault' AS vault_role, token_vault AS vault_acct, token_mint AS mint
    FROM alphabetic
    WHERE token_vault IS NOT NULL and token_vault != ''

    UNION ALL

    SELECT token_id, first_trade_time, 'quote_vault' AS vault_role, quote_vault AS vault_acct, wsol_mint AS mint
    FROM alphabetic, params
    WHERE quote_vault IS NOT NULL and quote_vault != ''
),

vault_transfers AS (
    SELECT
        f.token_id, f.first_trade_time, f.vault_role,
        t.block_slot, t.block_time, t.tx_id,
        t.tx_index, t.outer_instruction_index, t.inner_instruction_index,     

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
        token_id, tx_id,
        arbitrary(first_trade_time) as first_trade_time,
        MIN(block_slot) AS block_slot,
        MIN(block_time) AS block_time,

        MIN(COALESCE(tx_index, 0)) AS min_tx_index, -- ??????????????
        MIN(COALESCE(outer_instruction_index, 0)) AS min_outer_instruction_index,
        MIN(COALESCE(inner_instruction_index, 0)) AS min_inner_instruction_index,

        SUM(CASE WHEN vault_role = 'base_vault' THEN signed_amount_raw ELSE 0 END) AS delta_base_vault_raw,
        SUM(CASE WHEN vault_role = 'quote_vault' THEN signed_amount_raw ELSE 0 END) AS delta_quote_vault_raw
    FROM vault_transfers
    GROUP BY token_id, tx_id
),

ordered AS (
    SELECT
        token_id,

        row_number() OVER (
            PARTITION BY token_id
            -- Solana transactions order 
            ORDER BY block_slot, min_tx_index, min_outer_instruction_index, min_inner_instruction_index, tx_id
        ) AS event_seq,

        -- simplified timestamp: seconds since first trade
        CAST(to_unixtime(block_time) - to_unixtime(first_trade_time) AS BIGINT) AS t_rel_s,

        delta_base_vault_raw,
        delta_quote_vault_raw
    FROM tx_agg
)

-- compact repackaging of transactions
SELECT
    token_id,
    array_join(
        array_agg(
            CAST(event_seq AS VARCHAR) || ',' ||
            CAST(t_rel_s AS VARCHAR) || ',' ||
            CAST(delta_base_vault_raw AS VARCHAR) || ',' ||
            CAST(delta_quote_vault_raw AS VARCHAR)
            ORDER BY event_seq
        ),
        ';'
    ) AS packed_events
FROM ordered
GROUP BY token_id
-- absolute minimum activity for a token at launch
HAVING COUNT (*) > 30
