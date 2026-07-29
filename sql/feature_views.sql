-- Analyst-facing SQL views over the staged tables and the feature
-- snapshot. On Databricks these run as-is against the tables produced by
-- the pipeline (register the staged parquet paths as tables, or point a
-- catalogue schema at the staging prefix). In the test suite the same
-- statements execute against Spark temporary views, so every view here is
-- proven runnable, not decorative.
--
-- Grain warning that applies to every query in this file: customer_id is
-- unique per order in Olist. Person-level analysis must go through
-- customer_unique_id, which is why v_orders_enriched exists.

CREATE OR REPLACE TEMPORARY VIEW v_orders_enriched AS
SELECT
    o.order_id,
    c.customer_unique_id,
    c.customer_state,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    DATEDIFF(o.order_delivered_customer_date, o.order_estimated_delivery_date)
        AS delivery_delay_days,
    CASE
        WHEN o.order_delivered_customer_date IS NULL THEN NULL
        WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1
        ELSE 0
    END AS late_delivery
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;

-- Classic cohort retention matrix: customers grouped by first purchase
-- month, counted in each subsequent month they were active. Feeding a BI
-- tool exactly this shape is the usual first ask on a retention project.
CREATE OR REPLACE TEMPORARY VIEW v_monthly_cohort_retention AS
WITH first_purchase AS (
    SELECT
        customer_unique_id,
        MIN(order_purchase_timestamp) AS first_purchase_ts
    FROM v_orders_enriched
    WHERE order_status <> 'canceled'
    GROUP BY customer_unique_id
)
SELECT
    DATE_TRUNC('month', f.first_purchase_ts) AS cohort_month,
    CAST(MONTHS_BETWEEN(
        DATE_TRUNC('month', o.order_purchase_timestamp),
        DATE_TRUNC('month', f.first_purchase_ts)
    ) AS INT) AS months_since_first,
    COUNT(DISTINCT o.customer_unique_id) AS active_customers
FROM v_orders_enriched o
JOIN first_purchase f USING (customer_unique_id)
WHERE o.order_status <> 'canceled'
GROUP BY 1, 2;

CREATE OR REPLACE TEMPORARY VIEW v_delivery_performance_monthly AS
SELECT
    DATE_TRUNC('month', order_purchase_timestamp) AS order_month,
    COUNT(*) AS delivered_orders,
    ROUND(AVG(delivery_delay_days), 2) AS avg_delay_days,
    ROUND(AVG(CAST(late_delivery AS DOUBLE)), 4) AS late_rate
FROM v_orders_enriched
WHERE order_delivered_customer_date IS NOT NULL
GROUP BY 1;

-- Repeat rate by the first order's review score. NULL review_score is a
-- real segment (customers who never reviewed), not dirty data.
CREATE OR REPLACE TEMPORARY VIEW v_repeat_by_first_review AS
WITH ranked_orders AS (
    SELECT
        customer_unique_id,
        order_id,
        ROW_NUMBER() OVER (
            PARTITION BY customer_unique_id
            ORDER BY order_purchase_timestamp, order_id
        ) AS order_rank,
        COUNT(*) OVER (PARTITION BY customer_unique_id) AS n_orders
    FROM v_orders_enriched
    WHERE order_status <> 'canceled'
),
latest_review AS (
    SELECT
        order_id,
        review_score,
        ROW_NUMBER() OVER (
            PARTITION BY order_id
            ORDER BY review_creation_date DESC, review_id DESC
        ) AS review_rank
    FROM order_reviews
)
SELECT
    r.review_score,
    COUNT(*) AS customers,
    ROUND(AVG(CASE WHEN f.n_orders > 1 THEN 1.0 ELSE 0.0 END), 4) AS repeat_rate
FROM ranked_orders f
LEFT JOIN (
    SELECT order_id, review_score FROM latest_review WHERE review_rank = 1
) r ON f.order_id = r.order_id
WHERE f.order_rank = 1
GROUP BY r.review_score;

-- Summary of the model feature snapshot by region, the quick sanity view
-- a reviewer asks for before trusting any model built on the table.
CREATE OR REPLACE TEMPORARY VIEW v_feature_summary_by_region AS
SELECT
    customer_region,
    COUNT(*) AS customers,
    ROUND(AVG(CAST(label_repeat AS DOUBLE)), 4) AS repeat_rate,
    ROUND(AVG(monetary_total), 2) AS avg_monetary_total,
    ROUND(AVG(late_delivery_rate), 4) AS avg_late_delivery_rate,
    ROUND(AVG(avg_review_score), 3) AS avg_review_score
FROM customer_features
GROUP BY customer_region;
