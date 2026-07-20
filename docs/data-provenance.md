# Data Provenance

## Supported inputs

- CoinGecko historical price endpoints;
- yfinance as an educational/research fallback;
- wide-format CSV price data through the Python package.

CSV files should contain a date column and one numeric price column per asset.
Price inputs are preferred over externally calculated returns because the
platform can then apply a documented return convention consistently.

## Required publication metadata

Every published result should record:

- provider and endpoint or file name;
- extraction timestamp in UTC;
- requested and realized date ranges;
- quote currency;
- asset identifiers and symbols;
- number of raw and retained observations;
- missing-data and alignment treatment;
- core Simple-return convention plus any Advanced diagnostic Log convention;
- data frequency and calendar;
- configuration-file hash;
- software version or Git commit;
- random seed for simulated results.

## Mixed calendars

Crypto trades continuously while equities and exchange-traded products use
exchange calendars. The current cleaning pipeline forward-fills at most one
missing observation before dropping rows that remain incomplete. Depending on
the provider's combined index, this may retain some crypto-only calendar days
rather than removing every weekend observation. The realized common calendar
and the economic meaning of a "daily" observation must be reported in published
comparisons.

## Caching

CoinGecko payloads may be written to `data/cache/`. Caches are operational
artifacts and are ignored by Git. Do not publish a cache without confirming the
provider's redistribution terms and inspecting it for sensitive metadata.

## Data licensing

API accessibility does not automatically grant redistribution rights. Before a
sample dataset is committed, verify the provider's current terms. When rights
are uncertain, publish a download/reproduction script and metadata rather than
the vendor data itself.

## Private data

Never commit real holdings, transactions, account identifiers, client records,
monitoring databases, credentials, or proprietary signals. The public project
should demonstrate the workflow with public or synthetic inputs only.
