# Time Weighted Average [<sup><mark>experimental</mark></sup>](/extension/docs/README.md#tag-notes)

> [Description](#time-weighted-average-description)<br>
> [Example Usage](time-weighted-average-examples)<br>
> [API](#time-weighted-average-api) <br>
> [Notes on Parallelism and Ordering](#time-weight-ordering)<br>
> [Interpolation Methods Details](#time-weight-methods)<br>


## Description [](time-weighted-average-description)

Time weighted averages are commonly used in cases where a time series is not evenly sampled, so a traditional average will give misleading results. Consider a voltage sensor that sends readings once every 5 minutes or whenever the value changes by more than 1 V from the previous reading. If the results are generally stable, but with some quick moving transients, a simple average over all of the points will tend to over-weight the transients instead of the stable readings. A time weighted average weights each value by the duration over which it occured based on the points around it and produces correct results for unevenly spaced series. 

Timescale Analytics' time weighted average is implemented as an aggregate which weights each value either using a last observation carried forward (LOCF) approach or a linear interpolation approach ([see interpolation methods](#time-weight-methods)). While the aggregate is not parallelizable, it is supported with [continuous aggregation](https://docs.timescale.com/latest/using-timescaledb/continuous-aggregates). 

Additionally, [see the notes on parallelism and ordering](#time-weight-ordering) for a deeper dive into considerations for use with parallelism and some discussion of the internal data structures. 

---
## Example Usage [](time-weighted-average-examples)
For these examples we'll assume a table `foo` defined as follows:
```SQL
CREATE TABLE foo (
    measure_id      BIGINT,
    ts              TIMESTAMPTZ ,
    val             DOUBLE PRECISION,
    PRIMARY KEY (measure_id, ts)
);

```
Where the measure_id defines a series of related points. A simple use would be to calculate the time weighted average over the whole set of points for each `measure_id`. We'll use the LOCF method for weighting: 

```SQL
SELECT measure_id, 
    timescale_analytics_experimental.average(
        timescale_analytics_experimental.time_weight('LOCF', ts, val)
    )
FROM foo
GROUP BY measure_id;
```
(And of course a where clause can be used to limit the time period we are averaging, the measures we're using etc.).


We can also use the [`time_bucket` function](https://docs.timescale.com/latest/api#time_bucket) to produce a series averages in 15 minute buckets: 
```SQL
SELECT measure_id, 
    time_bucket('15 min'::interval, ts) as bucket, 
    timescale_analytics_experimental.average(
        timescale_analytics_experimental.time_weight('LOCF', ts, val)
    )
FROM foo
GROUP BY measure_id, time_bucket('15 min'::interval, ts);
```

Of course this might be more useful if we make a continuous aggregate out of it. We'll first have to make it a hypertable partitioned on the ts column, with a relatively large chunk_time_interval because the data isn't too high rate: 

```SQL
SELECT create_hypertable('foo', 'ts', chunk_time_interval=> '15 days'::interval, migrate_data => true);
```

Now we can make our continuous aggregate:

```SQL
CREATE MATERIALIZED VIEW foo_15 
WITH (timescaledb.continuous)
AS SELECT measure_id, 
    time_bucket('15 min'::interval, ts) as bucket, 
    timescale_analytics_experimental.time_weight('LOCF', ts, val)
FROM foo
GROUP BY measure_id, time_bucket('15 min'::interval, ts);
```

Note that here, we just use the `time_weight` function. It's often better to do that and simply run the `average` function when selecting from the view like so: 
```SQL
SELECT measure_id, 
    bucket, 
    timescale_analytics_experimental.average(time_weight)
FROM foo_15
```

It also allows us to re-aggregate from the continuous aggregate into a larger bucket size quite simply:

```SQL
SELECT measure_id, 
    time_bucket('1 day'::interval, bucket), timescale_analytics_experimental.average(
            timescale_analytics_experimental.time_weight(time_weight)
    )
FROM foo_15
GROUP BY measure_id, time_bucket('1 day'::interval, bucket);
```

We can also use this to speed up our initial calculation where we're only grouping by measure_id and producing a full average (assuming we have a fair number of points per 15 minute period):

```SQL
SELECT measure_id,
    timescale_analytics_experimental.average(
        timescale_analytics_experimental.time_weight(time_weight)
    )
FROM foo_15
GROUP BY measure_id
```
---

## Command List (A-Z) [](time-weighted-average-api)
> - [time_weight() (point form)](#time_weight_point)
> - [time_weight() (summary form)](#time-weight-summary) 
> - [average()](#time-weight-average)

---
## **time_weight() (point form)** [](time_weight_point)
```SQL
timescale_analytics_experimental.time_weight(
    method TEXT¹,
    ts TIMESTAMPTZ,
    value DOUBLE PRECISION
) RETURNS TimeWeightSummary
```
¹ Only two values are currently supported, 'linear' and 'LOCF', any capitalization of these will be accepted. [See interpolation methods for more info.](#time-weight-methods) 

An aggregate that produces a `TimeWeightSummary` from timestamps and associated values. 

### Required Arguments² [](time-weight-point-required-arguments)
|Name| Type |Description|
|---|---|---|
| `method` | `TEXT` | The weighting method we should use, options are 'linear' or 'LOCF', not case sensitive |
| `ts` | `TIMESTAMPTZ` |  The time at each point |
| `value` | `DOUBLE PRECISION` | The value at each point to use for the time weighted average|
<br>

² Note that `ts` and `value` can be `null`, however the aggregate is not evaluated on `null` values and will return `null`, but it will not error on `null` inputs.

### Returns

|Column|Type|Description|
|---|---|---|
| `time_weight` | `TimeWeightSummary` | A TimeWeightSummary object that can be passed to other functions within the time weighting API. |
<br>

### Sample Usage
```SQL
WITH t as (
    SELECT 
        time_bucket('1 day'::interval, ts) as dt,   
        timescale_analytics_experimental.time_weight('Linear', ts, val) AS tw -- get a time weight summary
    FROM foo
    WHERE id = 'bar'
    GROUP BY time_bucket('1 day'::interval, ts)
) 
SELECT 
    dt, 
    timescale_analytics_experimental.average(tw) -- extract the average from the time weight summary
FROM t;
```

## **time_weight() (summary form)** [](time-weight-summary)
```SQL
timescale_analytics_experimental.time_weight(
    tws TimeWeightSummary
) RETURNS TimeWeightSummary
```

An aggregate to compute a combined `TimeWeightSummary` from a series of non-overlapping `TimeWeightSummaries`. Non-disjoint `TimeWeightSummaries` will cause errors. See [Notes on Parallelism and Ordering](#time-weight-ordering) for more information.

### Required Arguments² [](time-weight-summary-required-arguments)
|Name| Type |Description|
|---|---|---|
| `tws` | `TimeWeightSummary` | The input TimeWeightSummary from a previous `time_weight` (point form) call, often from a [continuous aggregate](https://docs.timescale.com/latest/using-timescaledb/continuous-aggregates)|

### Returns

|Column|Type|Description|
|---|---|---|
| `time_weight` | `TimeWeightSummary` | A TimeWeightSummary object that can be passed to other functions within the time weighting API. |
<br>

### Sample Usage
```SQL
WITH t as (
    SELECT 
        date_trunc('day', ts) as dt,   
        timescale_analytics_experimental.time_weight('Linear', ts, val) AS tw -- get a time weight summary
    FROM foo
    WHERE id = 'bar'
    GROUP BY date_trunc('day')
), q as (
    SELECT timescale_analytics_experimental.time_weight(tw) AS full_tw -- do a second level of aggregation to get the full time weighted average
    FROM t
)
SELECT 
    dt, 
    timescale_analytics_experimental.average(tw),  -- extract the average from the time weight summary
    timescale_analytics_experimental.average(tw) / (SELECT full_tw FROM q LIMIT 1)  as normalized -- get the normalized average
FROM t;
```

## **average()** [](time-weight-average)
```SQL
timescale_analytics_experimental.average(
    tws TimeWeightSummary
) RETURNS DOUBLE PRECISION
```

A function to compute a time weighted average from a `TimeWeightSummary`.

### Required Arguments² [](time-weight-summary-required-arguments)
|Name| Type |Description|
|---|---|---|
| `tws` | `TimeWeightSummary` | The input TimeWeightSummary from a `time_weight` call.|

### Returns

|Column|Type|Description|
|---|---|---|
| `average` | `DOUBLE PRECISION` | The time weighted average computed from the `TimeWeightSummary`|
<br>

### Sample Usage

```SQL
SELECT 
    id,  
    timescale_analytics_experimental.average(tws) 
FROM (
    SELECT 
        id, 
        timescale_analytics_experimental.time_weight('LOCF', ts, val) AS tws 
    FROM foo 
    GROUP BY id
) t
```
---
## Notes on Parallelism and Ordering [](time-weight-ordering)

The time weighted average calculations we perform require a strict ordering of inputs and therefore the calculations are not parallelizable in the strict Postgres sense. This is because when Postgres does parallelism it hands out rows randomly, basically as it sees them to workers. However, if your parallelism can guarantee disjoint (in time) sets of rows, the algorithm can be parallelized, just so long as within some time range, all rows go to the same worker. This is the case for both [continuous aggregates](https://docs.timescale.com/latest/using-timescaledb/continuous-aggregates) and for [distributed hypertables](https://docs.timescale.com/latest/using-timescaledb/distributed-hypertables) (as long as the partitioning keys are in the group by, though the aggregate itself doesn't horribly make sense otherwise). 

We throw an error if there is an attempt to combine overlapping `TimeWeightSummaries`, for instance, in our example above, if you were to try to combine summaries across `measure_id`s it would error. This is because the interpolation techniques really only make sense within a given time series determined by a single `measure_id`. However, given that the time weighted average produced is a dimensionless quantity, a simple average of time weighted average should better represent the variation across devices, so the recommendation for things like baselines across many timeseries would be something like:

```SQL
WITH t as (SELECT measure_id, 
        timescale_analytics_experimental.average(
            timescale_analytics_experimental.time_weight('LOCF', ts, val)
        ) as time_weighted_average
    FROM foo
    GROUP BY measure_id)
SELECT avg(time_weighted_average) -- use the normal avg function to average our time weighted averages
FROM t;
```

Internally, the first and last points seen as well as the calculated weighted sum are stored in each `TimeWeightSummary` and used to combine with a neighboring `TimeWeightSummary` when re-aggregation or the Postgres `combine function` is called. In general, the functions support [partial aggregation](https://www.postgresql.org/docs/current/xaggr.html#XAGGR-PARTIAL-AGGREGATES) and partitionwise aggregation in the multinode context, but are not parallelizable (in the Postgres sense, which requires them to accept potentially overlapping input). 

Because they require ordered sets, the aggregates build up a buffer of input data, sort it and then perform the proper aggregation steps. In cases where memory is proving to be too small to build up a buffer of points causing OOMs or other issues, a multi-level aggregate can be useful. Following our example from above:

```SQL
WITH t as (SELECT measure_id, 
    time_bucket('1 day'::interval, ts),
    timescale_analytics_experimental.time_weight('LOCF', ts, val)
    FROM foo
    GROUP BY measure_id, time_bucket('1 day'::interval, ts)
    )
SELECT measure_id, 
    timescale_analytics_experimental.average(
        timescale_analytics_experimental.time_weight(time_weight)
    )
FROM t
GROUP BY measure_id;
```

Moving aggregate mode is not supported by `time_weight` and its use as a window function may be quite inefficient. 

---
## Interpolation Methods Details [](time-weight-methods)

Discrete time values don't always allow for an obvious calculation of the time weighted average. In order to calculate a time weighted average we need to choose how to weight each value. The two methods we currently use are last observation carried forward (LOCF) and linear interpolation. 

In the LOCF approach, the value is treated as if it remains constant until the next value is seen. The LOCF approach is commonly used when the sensor or measurement device sends measurement only when there is a change in value. 

The linear interpolation approach treats the values between any two measurements as if they lie on the line connecting the two measurements. The linear interpolation approach is used to account for irregularly sampled data where the sensor doesn't provide any guarantees 

Essentially, internally, the time weighted average computes a numerical approximation of the integral of the theoretical full time curve based on the discrete sampled points provided. We call this the weighted sum.  For LOCF, the the weighted sum will be equivalent to the area under a stepped curve:
```

|                        (pt 4)   
|          (pt 2)          *
|            *-------      |
|            |       |     | 
|(pt 1)      |       *------
|  *---------      (pt 3)  |
|  |                       |
|__|_______________________|______
             time
```
The linear interpolation is similar, except here it is more of a sawtooth curve. (And the points are different due to the limitations of the slopes of lines one can "draw" using ASCII art).
```
|                      (pt 4) 
|                        *      
|           (pt 2)     / |     
|            *       /   |
|          /   \   /     | 
|(pt 1)  /       *       | 
|      *      (pt 3)     |
|      |                 | 
|______|_________________|____________
             time
```

Here this ends up being equal to the rectangle with width equal to the duration between two points and height the midpoint between the two magnitudes. Once we have this weighted sum, we can divide by the total duration to get the time weighted average. 