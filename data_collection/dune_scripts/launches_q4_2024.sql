WITH params AS (
    SELECT
        TIMESTAMP '2024-10-01' AS q4_start,
        TIMESTAMP '2025-01-01' AS q4_end,
        'So11111111111111111111111111111111111111112' AS wsol_mint, -- WSOL mint
        -- Raydium program ids
        '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8' AS amm_v4,
        'CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C' AS cpmm
    ),

token_pair_2_pool as (
    SELECT
        min(token_bought_mint_address) as mintA_temp, -- guaranteed to differ, unless equal
        max(token_sold_mint_address) as mintB_temp,
        arbitrary(project) as dex,
        arbitrary(project_main_id) as pool_type,
        project_program_id as pool_id,
        MIN(block_time) as first_trade_time
    FROM dex_solana.trades, params
    WHERE
        project_program_id is not NULL and -- ignored for now, checked separately
        block_date < q4_end -- don't care about future; would cause leakage otherwise
    GROUP BY project_program_id
),

mints_identified as (
    SELECT
        dex, pool_type, pool_id, first_trade_time,
        least(mintA_temp, mintB_temp) as mintA, -- standardising mints A/B
        greatest(mintA_temp, mintB_temp) as mintB
    FROM token_pair_2_pool
),

token_2_pool as ( -- unpivot from token pairs to token-quote
    SELECT
        dex, pool_type, pool_id, first_trade_time,
        mintA as token_mint,
        mintB as quote_mint
    FROM mints_identified, params
    WHERE mintA != wsol_mint -- WSOL is only considered as quote

    UNION ALL

    SELECT
        dex, pool_type, pool_id, first_trade_time,
        mintB as token_mint,
        mintA as quote_mint
    FROM mints_identified, params
    WHERE mintB != wsol_mint
),

with_launch_time as (
    SELECT t.,
    MIN(
        CASE
            WHEN
                dex = 'raydium' -- only Raydium launches
                and pool_type IN (amm_v4, cpmm) -- only standard pool
                and quote_mint = wsol_mint -- only WSOL quote
            THEN first_trade_time
            ELSE NULL
        END
    ) OVER(PARTITION BY token_mint) as candidate_launch_time
    FROM token_2_pool t, params
),

time_and_existence_filtered as (
    SELECT t.
    FROM with_launch_time t, params
    WHERE
        candidate_launch_time is not NULL
        -- we only consider trading split between pool if the
        -- other pool creation time is in (-\infty, launch_t + 1 hour]
        and first_trade_time <= candidate_launch_time + INTERVAL '1' HOUR
        and candidate_launch_time >= q4_start
        and candidate_launch_time <= q4_end - INTERVAL '1' HOUR -- 1 hour buffer ("ripe" target)
),

counted as (
    SELECT *,
        COUNT(DISTINCT pool_id) OVER(PARTITION BY token_mint) as cnt_total,
        COUNT(DISTINCT (
            CASE
                WHEN dex != 'pumpdotfun' THEN pool_id
                ELSE NULL
            END
        )) OVER(PARTITION BY token_mint) as cnt_non_pumpdotfun
    FROM time_and_existence_filtered
),

only_valid as (
    SELECT
        token_mint, pool_type, pool_id, first_trade_time,
        (cnt_total > 1) as has_pumpdotfun_history
    FROM counted
    WHERE
        cnt_non_pumpdotfun = 1
        and cnt_total <= 2 -- removes weird cases, if exist
        and dex = 'raydium' -- extra check, though counters are sufficient
)

SELECT *
FROM only_valid
