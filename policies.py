from datetime import timedelta


def DetentionTank(sim, event):
    # A Detention tank is a tank that is intended to remain empty except for during periods of rain.
    # When it rains, the tank collects the stormwater and releases it at a set rate, in litres/hr.
    if (sim.Tank.volume/sim.Tank.capacity) >= sim.attributes['detention_levels'][event.month-1]:
        # Assumption is that demand interval is the smallest.
        return sim.attributes.detention_rate * sim.Tank.demand_interval
    else:
        return 0


def RobustSmartTank(sim, event):
    '''Robust Smart Tank function. Works the same as smart tank but can take
    a apply a robustnesss factor to the forecast. (essintally an over estimate).
    Can also take in a scrambled forecast. To get our orginial smart tank, use
    robustness factor of 1.'''
    if sim.use_forecast:  # Check to use fake forecast or perfect
        forecast_df = sim.forecast_events[(sim.rainfall_events.DateAndTime < (event + timedelta(days=1))) & (sim.rainfall_events.DateAndTime > (event))]
    else:
        forecast_df = sim.rainfall_events[(sim.rainfall_events.DateAndTime < (event + timedelta(days=1))) & (sim.rainfall_events.DateAndTime > (event))]

    if forecast_df.empty:  # If no rain forecast, no release
        return 0

    if sim.Tank.volume < (220*sim.Tank.occupancy):  # Have atleast a days supply of water in tank
        return 0
    else:
        predicted_rain_1day = forecast_df.Rainfall_mm.values.sum()*sim.Tank.area  # Rainforecast in litres
        expected_rain_1day = predicted_rain_1day*sim.robustness # Apply robustness to forecast

        if (expected_rain_1day + sim.Tank.volume) > sim.Tank.capacity:  # If tank will overflow
            # Calculate wanted spill
            out = ((sim.Tank.volume+expected_rain_1day)-sim.Tank.capacity)/(24*60/sim.Tank.demand_interval)
            # If spill is feasible, spill.
            if out <= sim.Tank.volume:
                return out
    return 0
