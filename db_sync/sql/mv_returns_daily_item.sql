DROP MATERIALIZED VIEW IF EXISTS mv_returns_daily_item;

CREATE MATERIALIZED VIEW mv_returns_daily_item AS
SELECT r.zid, r.itemcode, r.date, SUM(r.returnqty) AS returnqty
FROM (
    SELECT opcdt.zid, opcdt.xitem AS itemcode, opcrn.xdate AS date, opcdt.xqty AS returnqty
    FROM opcdt
    JOIN opcrn ON opcrn.xcrnnum::text = opcdt.xcrnnum::text AND opcrn.zid = opcdt.zid
    WHERE opcrn.xdate IS NOT NULL

    UNION ALL

    SELECT imtemptdt.zid, imtemptdt.xitem AS itemcode, imtemptrn.xdate AS date, imtemptdt.xqtyord AS returnqty
    FROM imtemptdt
    JOIN imtemptrn ON imtemptrn.ximtmptrn::text = imtemptdt.ximtmptrn::text AND imtemptrn.zid = imtemptdt.zid
    WHERE imtemptrn.xdate IS NOT NULL
) r
WHERE r.date IS NOT NULL
GROUP BY r.zid, r.itemcode, r.date;

CREATE INDEX ON mv_returns_daily_item (zid, itemcode, date);
