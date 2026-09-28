# Feature Availability at Forecast Origin

| Feature | Available? | Construction and control |
|---|---:|---|
| Demand lag 1 | Yes | Last actual or earlier recursive prediction |
| Demand lag 7 | Yes | Value seven days before forecast date |
| Demand lag 14 | Yes | Value fourteen days before forecast date |
| Demand lag 28 | Yes | Value twenty-eight days before forecast date |
| Rolling mean 7 | Yes | Mean of the seven values ending at t-1 |
| Rolling mean 28 | Yes | Mean of the twenty-eight values ending at t-1 |
| Category, store and state | Yes | Static series identifiers |
| Weekday and month | Yes | Known calendar fields |
| Event type | Yes | Known scheduled calendar field |
| State SNAP indicator | Yes | Known calendar field selected by series state |
| Actual test demand | No | Used only after prediction for evaluation |
| Future observed price | No | Excluded from the primary model |

