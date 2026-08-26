# Data directory

Large research catalogues are stored locally and are intentionally excluded
from Git. Preserve the following layout after obtaining the data from the
project's authorized shared storage.

```text
data/
├── raw/
│   └── red-sequence/
│       ├── redspell_highdens7_final.fits
│       └── redspell_highlum7_final.fits
└── processed/
    └── crossmatches/
        └── vega-ferrero/
            ├── match_VF_highdens.fits
            └── match_VF_highlum.fits
```

## Catalogue manifest

| File | Role | Rows | Columns |
| --- | --- | ---: | ---: |
| `raw/red-sequence/redspell_highdens7_final.fits` | High-density source catalogue | 1,482,074 | 171 |
| `raw/red-sequence/redspell_highlum7_final.fits` | High-luminosity source catalogue | 713,310 | 171 |
| `processed/crossmatches/vega-ferrero/match_VF_highdens.fits` | High-density catalogue matched to Vega-Ferrero morphology | 905,291 | 191 |
| `processed/crossmatches/vega-ferrero/match_VF_highlum.fits` | High-luminosity catalogue matched to Vega-Ferrero morphology | 34,768 | 191 |

The cross-match products contain the Vega-Ferrero morphology fields, including
`MP_LTG`, `MP_EdgeOn`, `FLAG_LTG`, `FLAG_EdgeOn`, and match `Separation`.
Analyses should initially use robust classifications: `FLAG_LTG == 4` for ETGs
and `FLAG_LTG == 5` for LTGs.
