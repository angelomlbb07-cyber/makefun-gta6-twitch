# GTA VI launch-day Twitch CCU forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/angelomlbb07-cyber/makefun-gta6-twitch/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/angelomlbb07-cyber/makefun-gta6-twitch/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day, hours 2–24, as five quantiles per hour, from
everything recorded up to the end of hour 1.

`main.py` (Python 3.13, standard library only) reads each case file, fits
two honest models per hour on the case's own comparable past launches —
a log-log OLS on hour-1 viewers and a two-feature ridge adding hour-1
channels — averages their medians, and takes the wider residual spread
(scaled 0.75) for the quantiles. See the docstring in `main.py` and
`MODEL.md` for the method and the validation numbers.

Run locally on the kit's practice cases:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
