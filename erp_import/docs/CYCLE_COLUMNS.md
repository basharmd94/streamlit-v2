# Online shop -> ERP 100007: column reference (up to `opmob`)

Full column lists for the tables the importer writes. The short version, the cycle and the live-server plan are in `CYCLE_MAP.md`.
**Written by:** *Importer* = my script. *ERP* = the ERP's own order-creation step. `-` = not written.

## Example order (used in every table)

Shop order **1000001** (internal id `order_01JBX3K8Q2`), cash on delivery, 5% order discount, delivery charge 100.

| Line | SKU | ERP item | Source (supplier) | Qty | Shop price | 100007 stock | PO quantity |
|---|---|---|---|---|---|---|---|
| 1 | 0718 | 01-00135 Hook--SS02 62**2mm | 100001 (SUP-000001) | 2 | 80 | 0 | 2, PO--000351, rate 57 |
| 2 | FZ000024 | 03-00033 Draino Powder (FZ-024) | 100005 (SUP-000003) | 5 | 125 | 3 | 2, PO--000352, rate 80 |

Items 160 + 625 = 785, discount 5% = 39.25, net 745.75, delivery 100.

---

## A. What the ERP does with an `opmob` row

Compared on every order the ERP created from the mobile app since March 2026: 11,884 in 100001, 6,853 in 100000 (rows joined through `opmob.xordernum`).

| `opmob` column | Where the ERP puts it | Matches |
|---|---|---|
| `invoiceno` | `opord.xref` and `opdor.xref`, as `Created From <invoiceno>` | 99% on the order; the delivery order's `xref` equals the order's in 100% |
| `xcus` | `opord.xcus`, `opdor.xcus` | 98% |
| `xemp` | `opord.xsp`, `opdor.xsp` | 97% |
| `xdate` | `opord.xdate` and again in `opord.xdatecuspo`; `opdor.xdate` is the delivery date | 97% (84% on the delivery order, it can be later) |
| `xordernum` | the ERP writes the new COMO number back into `opmob` | 100% |
| `xitem` | `opodt.xitem` + `xcode`, `opddt.xitem` + `xcode` | 99% |
| `xqty` | `opodt.xqtyreq` / `xqtyord`, `opddt.xqty` | 99.9% of matched lines |
| `xprice` | `opodt.xrate`, `opddt.xrate` | about 85% of lines; the rest were re-priced (mostly to `caitem.xstdprice`) |
| `xroword` | `opodt.xrow`, `opddt.xrow` | by construction |

**Not carried by the ERP:** `xcusname`, `xcusadd`, `xdesc`, `username`, `xterminal`, `xlat`, `xlong`, `xsl`, `invoicesl`, `xlinetotal`, `xtra1` to `xtra5`, `xappver`.
`xtra1/3/4/5` are empty in 100000, 100001 and 100005. `xtra2` is filled by the app everywhere, but the ERP does not copy it: the order total is recomputed from the lines (header = sum of its lines).
**The ERP fills from its own defaults:** `xdiv`, `xthana` (from the customer), `xproj`, `xwh`, status, currency, the creating user's email.
**Empty in every row of all four companies:** `opord.xcuspo`, `opdor.xcuspo`, `opcrn.xcuspo`. The line tables (`opodt`, `opddt`) have no reference column at all.

What this means:
1. **The shop order number reaches `opord` and `opdor` by itself if it is the `invoiceno`.** No change to your order-creation step. Lines link to their header by `xordernum` / `xdornum`.
2. The ERP builds an order from item, quantity, price, customer, salesman and date. Anything else I put in `opmob` (`xtra1` to `xtra5`, delivery charge, notes) stays in `opmob` unless your step is taught to copy it.
3. The ERP recomputes values from the lines it builds. `xtra2` and `xlinetotal` are not used, so a discount has to be inside `xprice` to reach the sales tables (question 2).

---

## B. Tables I touch

| Table | Access | Why |
|---|---|---|
| `opmob` | write, read, delete | the order, one row per line |
| `poord`, `poodt` | write, read, delete while `1-Open` | one open PO per order per source business |
| `caitem` | write, read | product sync; SKU lookup; source price |
| `xcodes` | write | price category row per item |
| `cacus`, `prmst` | read | UtilityMart and UTMART01 exist |
| `final_items_view` | read | stock sent to the shop |
| `imtrn`, `opdor` | read | 100007's own stock and open demand, for the PO shortfall |
| `um_*` (5 tables) | write | the importer's own bookkeeping |

---

## C. Columns

### C1. `opmob` (one row per order line; line 1 of the example shown)

| Column | Type | Value from the shop / rule | Example | Written by |
|---|---|---|---|---|
| `zid` | int | 100007 | 100007 | Importer |
| `ztime` | timestamp | time of insert | 2026-10-31 14:21 | Importer |
| `zutime` | timestamp | same as `ztime` | 2026-10-31 14:21 | Importer |
| `invoiceno` | varchar(100) | `UM-` + shop order number, same on every line of the order. **Proposed.** The ERP copies it into `opord.xref` and `opdor.xref` (section A) | UM-1000001 | Importer |
| `invoicesl` | bigint | the same order number as a plain number (for sorting); same on every line | 1000001 | Importer |
| `username` | varchar(100) | config `username` | utilitymart | Importer |
| `xemp` | varchar(100) | fixed employee (config `salesman`) | MR--000007 | Importer |
| `xcus` | varchar(100) | fixed customer UtilityMart (config) | CUS-000073 | Importer |
| `xcusname` | varchar(100) | fixed text | UtilityMart | Importer |
| `xcusadd` | varchar(100) | fixed text (the ERP does not carry it on) | UtilityMart online shop | Importer |
| `xitem` | varchar(100) | ERP item found from the shop SKU (`caitem.xalias`) | 01-00135 | Importer |
| `xdesc` | varchar(250) | `caitem.xdesc` of that item | Hook--SS02 62**2mm | Importer |
| `xqty` | int | line quantity | 2 | Importer |
| `xprice` | float | shop unit price before discount | 80 | Importer |
| `xstatusord` | varchar(100) | `New`. The ERP later changes it to `Order Created` or `Not enough stock to create Order` | New | Importer, then ERP |
| `xordernum` | varchar(100) | left empty; the ERP writes the new COMO number here | (blank), later COMO421005 | ERP (order creation) |
| `xroword` | int | line number 1..n | 1 | Importer |
| `xterminal` | varchar(10) | config `terminal` | UTM | Importer |
| `xdate` | date | order date (Dhaka time) | 2026-10-31 | Importer |
| `xsl` | varchar(100) | new uuid per line (primary key) | 5d0c1f3e-9a41-4b7e-8c2a-3f1e0b6d7a92 | Importer |
| `xlat` | float | left empty (the shop has no GPS) | (blank) | - |
| `xlong` | float | left empty | (blank) | - |
| `xlinetotal` | int | round(qty x price) before discount, whole taka | 160 | Importer |
| `xtra1` | int | order discount as a percent of the items total, whole number (exact value in `xtra5`). Column is integer | 5 | Importer |
| `xtra2` | float | order net total after discount, before delivery; same on every line | 745.75 | Importer |
| `xtra3` | varchar | shop internal order id; how the importer finds these rows again | order_01JBX3K8Q2 | Importer |
| `xtra4` | varchar | `cod`, or `online:<gateway>:<transaction id>` | cod | Importer |
| `xtra5` | varchar | JSON, no personal data: delivery charge, free-delivery flag, shipping method, order note, exact discount percent and amount | {"delivery_charge":100,"free_delivery":false,"shipping_method":"Pathao home delivery","note":"Call at the gate","discount_percent":5.0,"discount_amount":39.25} | Importer |
| `xappver` | numeric | not used | (blank) | - |

Also: read by `xtra3` to find an order's rows again; `xstatusord` and `xordernum` are read to report status to the shop (`New` = entered, `Order Created` + COMO number = order created). Rows are deleted on cancel and rebuilt on an order edit while still `New`.

### C2. `poord` (PO header; PO--000351 shown; one PO per order per source business)

| Column | Type | Value from the shop / rule | Example | Written by |
|---|---|---|---|---|
| `ztime` | timestamp | time of insert | 2026-10-31 14:21 | Importer |
| `zutime` | timestamp | left empty | (blank) | - |
| `zid` | int | 100007 | 100007 | Importer |
| `xpornum` | varchar(100) | next free PO number (max + 1) | PO--000351 | Importer |
| `xdate` | date | today | 2026-10-31 | Importer |
| `xsup` | varchar(100) | supplier of the SKU's source company: 100001 = SUP-000001, 100000 = SUP-000002, 100005 = SUP-000003 | SUP-000001 | Importer |
| `xsupref` | varchar(100) | shop order number | UM-1000001 | Importer |
| `xdatesupref` | date | today | 2026-10-31 | Importer |
| `xdiv` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xsec` | varchar(100) | template copy | Any | Importer (copied from the newest PO) |
| `xproj` | varchar(100) | config project | HMBR Online Shop | Importer |
| `xstatuspor` | varchar(100) | `1-Open` (staff confirm it later) | 1-Open | Importer |
| `xappamt` | numeric | 0 | 0 | Importer |
| `xpartial` | varchar(100) | `0` | 0 | Importer |
| `xcur` | varchar(100) | BDT | BDT | Importer |
| `xexch` | numeric | 1 | 1 | Importer |
| `xcoststat` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xrem` | varchar(500) | text with the order number and the shop internal id | Auto-created from online order UM-1000001 (order_01JBX3K8Q2) | Importer |
| `xwh` | varchar(100) | config warehouse | Sales Warehouse Online Shop | Importer |
| `xtempstr1` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xdisc` | numeric | 0 | 0 | Importer |
| `xdtwotax` | numeric | sum of the PO lines | 114 | Importer |
| `xdtdisc` | numeric | 0 | 0 | Importer |
| `xdttax` | numeric | 0 | 0 | Importer |
| `xval` | numeric | sum of the PO lines | 114 | Importer |
| `xdiscamt` | numeric | 0 | 0 | Importer |
| `xtotamt` | numeric | sum of the PO lines | 114 | Importer |
| `zemail` | varchar(50) | config `user_email` (optional) | (blank) | Importer |
| `xemail` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xtrnpor` | character(4) | `PO--` | PO-- | Importer |
| `xtypepor` | varchar(100) | `Local` | Local | Importer |
| `xwhoption` | varchar(100) | `Header` | Header | Importer |
| `xteam` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xmember` | varchar(100) | template copy | sobuj@hmbrbd.com | Importer (copied from the newest PO) |
| `xmanager` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xcounterno` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xgrnnum` | varchar(50) | left empty until the goods receipt | (blank) | - |
| `xref` | varchar(50) | shop order number | UM-1000001 | Importer |

### C3. `poodt` (PO line; PO--000351 line 1 shown)

| Column | Type | Value from the shop / rule | Example | Written by |
|---|---|---|---|---|
| `ztime` | timestamp | time of insert | 2026-10-31 14:21 | Importer |
| `zutime` | timestamp | left empty | (blank) | - |
| `zid` | int | 100007 | 100007 | Importer |
| `xpornum` | varchar(100) | the PO above | PO--000351 | Importer |
| `xrow` | int | line number 1..n | 1 | Importer |
| `xcode` | varchar(100) | = `xitem` | 01-00135 | Importer |
| `xcodebasis` | varchar(100) | `Our Code` | Our Code | Importer |
| `xitem` | varchar(100) | ERP item of the shop line | 01-00135 | Importer |
| `xlong` | varchar(1000) | blank | (blank) | Importer |
| `xstatuspdt` | varchar(100) | `Priced` | Priced | Importer |
| `xstype` | varchar(100) | `Stock-N-Sell` | Stock-N-Sell | Importer |
| `xwh` | varchar(100) | config warehouse | Sales Warehouse Online Shop | Importer |
| `xdropship` | character(6) | template copy | 0 | Importer (copied from the newest PO) |
| `xqtyord` | numeric | only the shortfall: ordered + other open web demand - 100007 stock - quantity on open POs. No shortfall, no PO line | 2 | Importer |
| `xqtygrn` | numeric | 0 (nothing received yet) | 0 | Importer |
| `xqtystk` | numeric | 0 | 0 | Importer |
| `xunitpur` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xcfpur` | numeric | 1 | 1 | Importer |
| `xwtunit` | numeric | template copy | 0 | Importer (copied from the newest PO) |
| `xunitwt` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xcur` | varchar(100) | template copy | BDT | Importer (copied from the newest PO) |
| `xexch` | numeric | template copy | 1 | Importer (copied from the newest PO) |
| `xrate` | numeric | the source company's `caitem.xstdprice` for the item (100001, item 0718) | 57 | Importer |
| `xmargin` | numeric | template copy | 0 | Importer (copied from the newest PO) |
| `xdisc` | numeric | 0 | 0 | Importer |
| `xdiscf` | numeric | template copy | 0 | Importer (copied from the newest PO) |
| `xcomm` | numeric | template copy | 0 | Importer (copied from the newest PO) |
| `xpricebasis` | varchar(100) | `Entered` | Entered | Importer |
| `xcurprice` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xexchbuy` | numeric | template copy | 0 | Importer (copied from the newest PO) |
| `xprice` | numeric | = `xrate` | 57 | Importer |
| `xdatesch` | date | today. **Real PO lines hold 2999-12-31**; I will change this to match | 2026-10-31 | Importer |
| `xtaxcat` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xtaxcode1` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xtaxrate1` | numeric | template copy | 0 | Importer (copied from the newest PO) |
| `xtaxcode2` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xtaxrate2` | numeric | template copy | 0 | Importer (copied from the newest PO) |
| `xtaxcode3` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xtaxrate3` | numeric | template copy | 0 | Importer (copied from the newest PO) |
| `xtaxcode4` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xtaxrate4` | numeric | template copy | 0 | Importer (copied from the newest PO) |
| `xtaxcode5` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xtaxrate5` | numeric | template copy | 0 | Importer (copied from the newest PO) |
| `xlandcost` | numeric | template copy | 0 | Importer (copied from the newest PO) |
| `xdttax` | numeric | 0 | 0 | Importer |
| `xdtdisc` | numeric | 0 | 0 | Importer |
| `xdtwotax` | numeric | qty x rate | 114 | Importer |
| `xlineamt` | numeric | qty x rate | 114 | Importer |
| `xpoqot` | varchar(100) | template copy | (blank) | Importer (copied from the newest PO) |
| `xline` | int | 0 | 0 | Importer |
| `xbatch` | varchar(200) | template copy | (blank) | Importer (copied from the newest PO) |
| `xcompwh` | varchar(100) | template copy | Sales Warehouse Online Shop | Importer (copied from the newest PO) |
| `xdescdet` | varchar(250) | template copy | (blank) | Importer (copied from the newest PO) |

### C4. `caitem` (zid 100007; product sync; example is the existing item 01-00135, new-item values noted)

For an **existing** item only `xdesc`, `xlong`, `xnote`, `xlinks`, `xscode` (if empty), `xstdprice` (raise only) and `zutime` change. Every other column is written once, when the item is first created.

| Column | Type | Value from the shop / rule | Example | Written by |
|---|---|---|---|---|
| `ztime` | timestamp | time of insert | 2026-10-31 14:20 | Importer (new item only) |
| `zutime` | timestamp | set on every update | 2026-10-31 14:20 | Importer (update) |
| `zid` | int | 100007 | 100007 | Importer (new item only) |
| `xitem` | varchar(100) | next free code for the source business: HMBR `01-`, GI `02-`, ZEPTO `03-`, FIXIT `04-`, OUTSIDE `05-` | 01-00135 (existing; a new item gets the next free code) | Importer (new item only) |
| `xalias` | varchar(100) | shop SKU | 0718 | Importer (new item only) |
| `xitemnew` | varchar(100) | source company id, from the shop's `erp_source` | 100001 | Importer (new item only) |
| `xitemold` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xdrawing` | varchar(100) | left empty | (blank) | Importer (new item only) |
| `xscode` | varchar(100) | VG code: reused from the product handle, else next `VG####`; none for a single-variant product | VG0094 | Importer (new, or update if empty) |
| `xdesc` | varchar(250) | shop variant name | Hook--SS02 62**2mm | Importer (new + update) |
| `xlong` | varchar(1000) | shop product title | SS Hook | Importer (new + update) |
| `xlinks` | varchar(500) | shop product handle | ss-hook | Importer (new + update) |
| `xgitem` | varchar(100) | by source: Trading Item / GI (Home Product) Item / Zepto Item / Fixit Item / Outside Item | Trading Item | Importer (new item only) |
| `xcitem` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xcat` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xpricecat` | varchar(100) | = SKU (key to the price category row) | 0718 | Importer (new item only) |
| `xtaxcat` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xduty` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xorigin` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xdiv` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xwh` | varchar(100) | config warehouse | Sales Warehouse Online Shop | Importer (new item only) |
| `xsup` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xtypeserial` | varchar(100) | template copy | 0-None | Importer (new item only, copied from template) |
| `xbatchman` | varchar(1) | template copy | 0 | Importer (new item only, copied from template) |
| `xabc` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xlife` | int | template copy | 0 | Importer (new item only, copied from template) |
| `xwtunit` | numeric | template copy | 0 | Importer (new item only, copied from template) |
| `xunitwt` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xminordqty` | numeric | template copy | 0 | Importer (new item only, copied from template) |
| `xminordval` | numeric | template copy | 0 | Importer (new item only, copied from template) |
| `xordmult` | numeric | template copy | 0 | Importer (new item only, copied from template) |
| `xyield` | numeric | template copy | 100 | Importer (new item only, copied from template) |
| `xdtf` | int | template copy | 0 | Importer (new item only, copied from template) |
| `xptf` | int | template copy | 0 | Importer (new item only, copied from template) |
| `xleadf` | int | template copy | 0 | Importer (new item only, copied from template) |
| `xleadv` | int | template copy | 0 | Importer (new item only, copied from template) |
| `xleadt` | int | template copy | 0 | Importer (new item only, copied from template) |
| `xstdcost` | numeric | left empty | (blank) | Importer (new item only) |
| `xstdprice` | numeric | shop price. On update it is only raised, never lowered | 80 | Importer (new + raise only) |
| `xunitstk` | varchar(100) | template copy | Pcs | Importer (new item only, copied from template) |
| `xunitalt` | varchar(100) | template copy | Pcs | Importer (new item only, copied from template) |
| `xunitiss` | varchar(100) | template copy | Pcs | Importer (new item only, copied from template) |
| `xunitpck` | varchar(100) | template copy | Pcs | Importer (new item only, copied from template) |
| `xunitsta` | varchar(100) | template copy | Pcs | Importer (new item only, copied from template) |
| `xcfiss` | numeric | template copy | 1 | Importer (new item only, copied from template) |
| `xcfpck` | numeric | template copy | 1 | Importer (new item only, copied from template) |
| `xcfsta` | numeric | template copy | 1 | Importer (new item only, copied from template) |
| `xtypestk` | varchar(100) | template copy | Stock-N-Sell | Importer (new item only, copied from template) |
| `xhide` | character(1) | template copy | 0 | Importer (new item only, copied from template) |
| `xwtnet` | numeric | template copy | 0 | Importer (new item only, copied from template) |
| `xunitlen` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xl` | numeric | template copy | 0 | Importer (new item only, copied from template) |
| `xw` | numeric | template copy | 0 | Importer (new item only, copied from template) |
| `xh` | numeric | template copy | 0 | Importer (new item only, copied from template) |
| `xdateeff` | date | template copy | 2026-10-01 | Importer (new item only, copied from template) |
| `xdateexp` | date | template copy | 2056-10-01 | Importer (new item only, copied from template) |
| `xnote` | varchar(1000) | shop variant options | Size:62mm,Thickness:2mm,Design:Plain Round | Importer (new + update) |
| `xexcisecat` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xbrand` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xstoporder` | varchar(100) | template copy | 0 | Importer (new item only, copied from template) |
| `xsplprice` | numeric | template copy | (blank) | Importer (new item only, copied from template) |
| `xcatful` | varchar(100) | `Auto` | Auto | Importer (new item only) |
| `xmatrix` | varchar(1) | template copy | 0 | Importer (new item only, copied from template) |
| `xcutpcs` | varchar(1) | template copy | 0 | Importer (new item only, copied from template) |
| `xtypedim` | varchar(100) | template copy | Extension | Importer (new item only, copied from template) |
| `xminprice` | numeric | template copy | 0 | Importer (new item only, copied from template) |
| `xmargincost` | numeric | template copy | 0 | Importer (new item only, copied from template) |
| `xbarcode` | varchar(100) | left empty | (blank) | Importer (new item only) |
| `xmultisel` | numeric | template copy | 1 | Importer (new item only, copied from template) |
| `xmultipur` | numeric | template copy | 1 | Importer (new item only, copied from template) |
| `xstrip` | numeric | template copy | 0 | Importer (new item only, copied from template) |
| `xloc` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xcurcost` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xcurprice` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xdisc` | numeric | template copy | (blank) | Importer (new item only, copied from template) |
| `xcomm` | numeric | template copy | (blank) | Importer (new item only, copied from template) |
| `xstype` | varchar(100) | template copy | Stock-N-Sell | Importer (new item only, copied from template) |
| `xunitpur` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xunitsel` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xcfpur` | numeric | template copy | (blank) | Importer (new item only, copied from template) |
| `xcfsel` | numeric | template copy | (blank) | Importer (new item only, copied from template) |
| `zemail` | varchar(50) | template copy | (blank) | Importer (new item only, copied from template) |
| `xemail` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xbin` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xmanufacturer` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xtitem` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xwarranty` | int | template copy | (blank) | Importer (new item only, copied from template) |
| `xmethserial` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xcostmeth` | varchar(100) | template copy | Weighted Average | Importer (new item only, copied from template) |
| `xbinman` | varchar(1) | template copy | (blank) | Importer (new item only, copied from template) |
| `xratio` | numeric | template copy | (blank) | Importer (new item only, copied from template) |
| `xqtydor` | numeric | template copy | (blank) | Importer (new item only, copied from template) |
| `xqtyrem` | numeric | template copy | (blank) | Importer (new item only, copied from template) |
| `xnetrate` | numeric | template copy | (blank) | Importer (new item only, copied from template) |
| `xremrate` | numeric | template copy | (blank) | Importer (new item only, copied from template) |
| `xremamt` | numeric | template copy | (blank) | Importer (new item only, copied from template) |
| `xflagrfq` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xdateacq` | date | template copy | (blank) | Importer (new item only, copied from template) |
| `xdatearr` | date | template copy | (blank) | Importer (new item only, copied from template) |
| `xsrate` | numeric | template copy | (blank) | Importer (new item only, copied from template) |
| `xcolor` | varchar(100) | template copy | (blank) | Importer (new item only, copied from template) |
| `xmaterial` | varchar(150) | template copy | (blank) | Importer (new item only, copied from template) |

### C5. `xcodes` (price category row, one per item)

| Column | Type | Value from the shop / rule | Example | Written by |
|---|---|---|---|---|
| `ztime` | timestamp | time of insert | 2026-10-31 14:20 | Importer |
| `zutime` | timestamp | set when `zactive` changes | (blank) | Importer |
| `zid` | int | 100007 | 100007 | Importer |
| `xtype` | varchar(100) | `Price Category` | Price Category | Importer |
| `xcode` | varchar(100) | = `caitem.xitem` | 01-00135 | Importer |
| `xdescdet` | varchar(250) | not written today; existing rows hold the item description. **I will fill it with `caitem.xdesc`** | Hook--SS02 62**2mm | Importer (proposed) |
| `xprops` | varchar(1000) | - | (blank) | - |
| `xcodealt` | varchar(100) | shop SKU | 0718 | Importer |
| `xteam` | varchar(100) | - | (blank) | - |
| `zactive` | varchar(1) | `1` if the shop item is active, else `0` | 1 | Importer |

### C6. Tables I only read

| Table | Columns read | What for | Example |
|---|---|---|---|
| `cacus` (100007) | `xcus`, `xorg` | check UtilityMart exists | CUS-0000xx |
| `prmst` (100007) | `xemp` | check UTMART01 exists | UTMART01 |
| `caitem` (source company) | `xitem`, `xstdprice` | PO rate; the item is 100007's `xalias` | 100001, 0718, 57 |
| `final_items_view` | `zid`, `item_id`, `stock` | stock for the shop, per SKU (`zid` = `xitemnew`, `item_id` = `xalias`); floored at 0; changed SKUs only, full list daily | 100001, 0718, 4190 |
| `imtrn` (100007) | `xitem`, `xqty`, `xsign` | 100007's own stock for the shortfall | 03-00033, 3 |
| `poord`, `poodt` | `xpornum`, `xstatuspor`, `xitem`, `xqtyord`, `xqtygrn` | quantity already on order | |
| `opdor` | `zid`, `xordernum` | an order already delivered no longer counts as open demand | |

### C7. The importer's own tables

| Table | Columns | Purpose |
|---|---|---|
| `um_sync_state` | `id, last_seq, last_full_stock_date, last_full_status_date, updated_at` | where I am in the shop's event feed |
| `um_processed_events` | `event_id, seq, type, key, status, result, processed_at` | every shop event once, with its result |
| `um_order_map` | `external_order_id, display_id, invoiceno, invoicesl, customer_code, version, state, po_numbers, last_status, last_status_at, info, created_at, updated_at` | shop order -> `opmob` invoice -> PO numbers (this is the order-ID tracking for each PO) |
| `um_stock_sent` | `sku, on_hand, sent_at` | last stock value sent per SKU |
| `um_attention` | `id, created_at, kind, order_ref, details, resolved` | staff to-do list (manual sourcing, locked orders) |
| `um_customer_map` | | no longer used |

---

## D. Notes

- `xtra1` is an integer column in the local copy: the discount is a whole percent there, the exact figure is in `xtra5`.
- `xtra1/3/4/5` stay in `opmob` unless your order-creation step copies them. The order number travels on `invoiceno`.
