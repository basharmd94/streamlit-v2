# Sales-Related Reports — Quick Reference

For WhatsApp outreach planning. Each report's WhatsApp/Mobile column comes from `cacus.xtaxnum` (whatsapp) and `cacus.xmobile` (mobile) — present wherever a customer-level table is shown.

## Target Management → Individual Salesman (default view)
Per-salesman, per-customer **ordered vs. not-ordered this month** breakdown, with WhatsApp number per customer.
- **🔴 Not Ordered**: customers assigned to a salesman who haven't bought this month — flags pending mobile orders (`opmob`) separately. Best list for "haven't heard from you" nudges.
- **✅ Ordered**: customers who already bought this month.
- **📦 Customer-Product Breakdown**: monthly qty/value pivot per customer × product — good for "you usually buy X, restock?" messages.
- **🚫 Customers with No Sales**: customers in the directory (by area) with zero sales in the loaded window at all — cold/dormant list.

## Target Management → All Salesmen Overview / Salesman Score
Salesman-level KPI summary (sales vs target, collection %, unique customers/products, AR balance, composite score). Useful for deciding *which salesman's book* to prioritize for a campaign, not for direct customer targeting.

## Target Management → Moving Average
3-month trailing average sales by Salesman / Product / Product Group. Good for spotting declining trends to target with win-back messaging.

## Customer Data View (Overall Sales Analysis page)
Drill-down: pick one customer (+ optional salesman/product/area) → full transaction history (sales & returns). Good for a 1:1 "here's your order history" lookup before messaging a specific customer.

## Overall Sales Analysis
ZID-wide sales trends: YoY, month-vs-month, distribution by product/area/salesman, summary stats. Aggregate only — no per-customer contact info. Useful for picking *which products/areas* to push in a campaign.

## Basket Analysis
What gets bought together.
- **Customer Pattern**: per-customer product affinity / repeat-purchase pattern.
- **Area Pattern**: product affinity by city/area.
- **Product / Product Group Basket**: co-purchase pairs ("customers who bought X also bought Y") — good for cross-sell message content.
- **Shipment Impact**: which items move after a new shipment lands — good timing signal for "just restocked" messages.

## Daily Sales Analysis
Day-by-day sales pivot + moving-average benchmark, by salesman/product/area. Good for short-term momentum checks, not customer-level targeting.

## Collection Analysis
- **Salesman Due / Customer Ledger / AR Customer Ledger**: per-customer outstanding balance (with WhatsApp number) — the list for payment-reminder messages.
- **Collection Performance**: collection vs. sales %, YoY/month comparisons — aggregate.

## AR Analysis
Customer-level AR ledger with month-by-month due breakdown (by origin month) and till-date balance, filterable by salesman. Same underlying balance logic as Salesman Due — alternate cut for due-reminder lists.

## Inventory / Current Stock (Target Management)
Current on-hand stock by item — use to check availability before promoting a product in a campaign, not customer-facing.

---

### Quick pick for WhatsApp campaigns
| Goal | Report |
|---|---|
| Re-engage lapsed/no-order customers | Individual Salesman → Not Ordered / No Sales |
| Cross-sell / restock nudge | Customer-Product Breakdown, Basket Analysis |
| Payment reminders | Collection Analysis (Salesman Due) or AR Analysis |
| New-shipment promo | Basket Analysis → Shipment Impact |
| Pick target products/areas | Overall Sales Analysis |
