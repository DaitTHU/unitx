from datetime import timedelta
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx import Quantity, time
from unitx.exceptions import DimensionError
from unitx.factor import Factor
from unitx.unit import ScaledUnit, Unit


@pytest.mark.parametrize(
    'name', ['time', 'monotonic', 'perf_counter', 'process_time', 'thread_time']
)
def test_time_clock_exact(monkeypatch, name):
    # Large timestamps must reach the Quantity without an intermediate float.
    ticks = iter([2**63 + 1, 2**63 + 4])
    monkeypatch.setattr(time._time, name + '_ns', lambda: next(ticks), raising=False)
    start = getattr(time, name)()
    stop = getattr(time, name)()
    assert type(start.value) is int
    assert start.value == 2**63 + 1
    assert start.unit == Unit('ns')
    assert (stop - start) == Quantity(3, 'ns')
    assert start.value_in('s') == Fraction(2**63 + 1, 10**9)


def test_time_live_clocks():
    assert type(time.time().value) is int
    for clock in (time.monotonic, time.perf_counter, time.process_time):
        assert clock() <= clock()


@pytest.mark.parametrize('duration', [timedelta.min, timedelta.max, timedelta(microseconds=-1)])
def test_timedelta_exact_roundtrip(duration):
    q = time.from_timedelta(duration)
    assert type(q.value) is int
    assert q.unit == Unit('us')
    assert time.to_timedelta(q) == duration


@pytest.mark.parametrize('nanoseconds, microseconds', [(500, 0), (1500, 2), (2500, 2), (-1500, -2)])
def test_to_timedelta_rounding(nanoseconds, microseconds):
    assert time.to_timedelta(Quantity(nanoseconds, 'ns')) == timedelta(microseconds=microseconds)


def test_to_timedelta_decimal_context():
    with localcontext() as context:
        context.prec = 2
        assert time.to_timedelta(Quantity(Decimal('1.2345675'), 's')) == timedelta(
            microseconds=1234568
        )
    unit = ScaledUnit('s', Factor(Fraction(1, 3)))
    assert time.to_timedelta(Quantity(3, unit)) == timedelta(seconds=1)
    assert time.to_timedelta(Quantity(Fraction(1, 3), 'min')) == timedelta(seconds=20)


def test_to_timedelta_invalid_range():
    with pytest.raises(OverflowError):
        time.to_timedelta(time.from_timedelta(timedelta.max) + Quantity(1, 'us'))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), Decimal('-Infinity')])
def test_time_invalid_nonfinite(value):
    for function in (time.to_timedelta, time.sleep):
        with pytest.raises(ValueError, match='finite'):
            function(Quantity(value, 's'))


@pytest.mark.parametrize('value', [True, 1j, object()])
def test_time_invalid_magnitude(value):
    for function in (time.to_timedelta, time.sleep):
        with pytest.raises(TypeError, match='magnitude'):
            function(Quantity(value, 's'))


def test_time_invalid_dimension():
    for function in (time.to_timedelta, time.sleep):
        with pytest.raises(DimensionError):
            function(Quantity(1, 'm'))


def test_time_invalid_duration():
    for function in (time.to_timedelta, time.sleep):
        with pytest.raises(TypeError, match='duration Quantity'):
            function(1)
    with pytest.raises(TypeError, match='datetime.timedelta'):
        time.from_timedelta(1)


def test_sleep_conversion(monkeypatch):
    calls = []
    monkeypatch.setattr(time._time, 'sleep', calls.append)
    time.sleep(Quantity(125, 'ms'))
    time.sleep(Quantity(Decimal('0'), 's'))
    assert calls == [0.125, 0.0]
    with pytest.raises(ValueError, match='nonnegative'):
        time.sleep(Quantity(Fraction(-1, 10**400), 's'))
    assert len(calls) == 2
