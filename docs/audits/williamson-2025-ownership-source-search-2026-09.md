# Williamson 2025 ownership source located: September 7, 2026

**Subsequent implementation:** the approved integration is documented in the
[reconciliation and rerun audit](williamson-2025-ownership-integration-2026-09.md).
Statements below about sources not yet being used describe the source-search
turn, before that integration.

## Correction to the earlier search conclusion

The 2025 Williamson certified ownership roll **is publicly downloadable**.
The fresh search located it on WCAD's official Historical Data page and verified
the archive bytes. The previous search failed to locate the source; its absence
from our cached inputs should not have been treated as evidence that it was
unavailable online.

The older cached `wcad_owners.csv` remains a 2026 owner extract. Finding the
correct source does not validate the earlier code's practice of labeling that
current extract as 2025. The newly found file is a different historical source.

## Verified official source

- [WCAD Historical Data landing page](https://www.wcad.org/historical-data/).
- [Public 2025 Certified Appraisal Roll download](https://williamsoncad.sharepoint.com/:u:/s/OperationsInfo/IQCilG6icMdMT4rDZYoXvQXGAZ2wIl2IOZiksNZuKN9Nqn4?e=OCQkek).
- Landing-page description: appraisal roll report as of certification; listed
  size 55 MB, posted July 16, 2026.
- Hosting: public SharePoint `OperationsInfo` site, under
  `Shared Documents/HistoricalData/2025/Certification-Report-2025.zip`.
- HTTP response: 200, `application/x-zip-compressed`; Last-Modified
  July 16, 2026, 17:21:38 GMT.
- ZIP size: **57,459,889 bytes**.
- SHA-256: `279692f8bb354cde93eb2e71cdd500d9ae7b598a19e40ba9d281cef6e6d96483`.
- ZIP member: `Certification-Report-2025.txt`, **1,381,824,532 bytes**;
  member timestamp July 11, 2025, 19:04 (timezone unspecified).
- Internal report header: `2025 CERTIFIED ROLL PUBLIC APPROVED`;
  report timestamp July 11, 2025, 6:37 PM (timezone unspecified).
- The archive passes `unzip -t` integrity validation.

The downloaded report contains parcel/QuickRef IDs, displayed owner IDs and
names, mailing addresses, situs addresses, exemption codes, and appraisal
values. It is not the property-only 2025 table examined in the earlier search.
The internal year is verified separately from the 2026 web posting date.

An immutable research copy is saved outside the pipeline's automatically
consumed paths, at:

`data/raw_parcels/ownership_research/williamson_2025_certified/Certification-Report-2025.zip`

This location is Git-ignored. The research retrieval was completed September 8,
2026 UTC (September 7 in Austin). No account login or records request was needed;
the public download uses an anonymous session cookie. Session cookies and
cookie-bearing HTTP headers are not copied into the repository.

## Why the earlier search missed it

The known 2024 report uses a WordPress uploads ZIP URL. The earlier search
checked analogous 2025 upload paths, the live Socrata catalog, web archives,
and the taxing-entity report portal. The actual 2025 file is hosted on a
different SharePoint site and linked through an image-only download control on
the official Historical Data page. The text-rendered page lists the report
but does not expose that image link as a numbered text hyperlink. Reading the
page's HTML revealed the direct public share.

This explains the retrieval route, not an excuse for the missed source. The
page lists a July 2026 posting date, so we should not describe the file as newly
published in response to this search. The earlier unsuccessful search was not
exhaustive enough to support treating a public-information request as necessary.

## Scope and remaining checks

### Certified-roll parcel coverage

The existing read-only, year-checking adapter parsed 290,394 report owner
records and matched **13,179 of 13,626 Williamson target parcels (96.72%)**.
Those matches cover **97.67% of validated target units**. On the actual mapped
H3 support, 12,593 of 13,014 parcels match, covering **97.73% of units**.
There are 447 absent target IDs in the full target. Among matched records, 4,763 displayed owner
names are potentially clipped. With the existing conservative classifier,
corporate status is known for 12,029 targets and financialized status for
8,512; 8,259 targets have all three flags known. Matching a source parcel is
therefore not the same as having complete ownership classification evidence.
On mapped H3 support, corporate status is known for 85.61% of units and
financialized status for 69.06% of units. The longer-name GIS source below is
therefore a useful candidate supplement, subject to owner/date reconciliation.

### Additional public July 2025 parcel source

A second verified source supplies substantially longer ownership/address
fields and may help resolve the printed roll's clipping limitations:

- [TxDOT's public 2025 Land Parcels item](https://www.arcgis.com/home/item.html?id=bfee1546d60b4a998ad37a8765941898),
  representing TxGIO's 2025 statewide parcel collection.
- [Williamson July 2025 county ZIP, public TNRIS/TxGIO S3 origin](https://s3.amazonaws.com/data.tnris.org/0fa04328-872e-481c-b453-126a74777593/resources/stratmap25-landparcels_48491_lp.zip).
- County archive size: **126,154,817 bytes**; SHA-256:
  `f3f086444b0936f8b06fa9a6b2e47dc42e16737ed7a9ae5278c179056cc0f75c`.
- File geodatabase: `stratmap25-landparcels_48491_williamson_202507.gdb`;
  282,983 feature rows and 37 fields. All county rows have `TAX_YEAR = 2025`
  and `DATE_ACQ = 20250701`.
- Metadata identifies July 2025 acquisition and July 28, 2025 processing
  directly from the WCAD parcel database view. The public S3 object's
  Last-Modified date is September 4, 2025.
- Fields include `Prop_ID`, `OWNER_NAME`, `MAIL_ADDR`, mailing components,
  situs components, source and date/year fields. No homestead field is present.
- **All 13,626 target parcel IDs match**, represented by 13,738 feature rows;
  112 IDs have duplicate features. Their owner names and combined mailing
  addresses agree. Deduplicate to the fixed EWS parcel ID before weighting units.
- All matching rows have nonblank owner names, mailing address, and mailing
  street, but **65 unique parcels contain `UNAVAILABLE` placeholders** and
  must not be treated as valid negative ownership evidence. The remaining
  13,561 target names are nonblank/nonplaceholder. Eighteen international
  address rows lack separate mailing city/state/ZIP fields.
- Among unique target parcels, 4,102 names exceed 30 characters, 996 exceed 50,
  and 73 are exactly 80. The maximum observed length is 80 despite the field's
  254-character capacity, and some names may still be clipped. This is longer
  evidence, not a guarantee of complete names or a complete co-owner roster.

A Git-ignored research copy is preserved at:

`data/raw_parcels/ownership_research/williamson_2025_txgio/stratmap25-landparcels_48491_lp.zip`

Aggregate-only `verification.json` files are saved beside both research ZIPs,
recording source checks, target matches and remaining evidence limitations.

The two sources have different snapshot/processing semantics. Before combining
them, reconcile parcel identity, repeated features, owner-name agreement and
source dates. Do not automatically attach certified-roll homestead evidence
to a different owner observed in the GIS extract. Neither source has been
substituted into the current comparison pipeline.

### Implementation boundary

This turn locates, preserves, and validates the source. It does not replace
canonical inputs, modify the classifier, or rerun the ownership comparison or
clusters. The prior Part 2 run still reports missing Williamson 2025 because
that run predates this discovery.

The next implementation step is to record the correct SharePoint provenance,
adapt the report through the same year-checked classifier, and recompute
two-year parcel/unit coverage. Printed names and addresses may still be
truncated, as in 2024; the displayed owner is not a verified complete co-owner
roster. Source availability removes the missing-file blocker, but does not
automatically guarantee sufficient classification coverage for every hex.

Its 2025 internal certification date supports the retrospective appraisal-vintage
comparison. The 2026 posting date should remain explicit; this download alone
does not establish what was publicly available by a historical analysis cutoff.
