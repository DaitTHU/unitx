from datetime import datetime, timedelta, timezone, tzinfo
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx import AffineQuantity, AffineUnit, Quantity, time
from unitx.exceptions import DimensionError
from unitx.factor import Factor
from unitx.unit import ScaledUnit, Unit


@pytest.mark.parametrize(
    'name', ['time', 'monotonic', 'perf_counter', 'process_time', 'thread_time']
)
def test_time_clock_exact(monkeypatch, name):
    # Large timestamps must reach the point without an intermediate float.
    ticks = iter([2**63 + 1, 2**63 + 4])
    monkeypatch.setattr(time._time, name + '_ns', lambda: next(ticks), raising=False)
    start = getattr(time, name)()
    stop = getattr(time, name)()
    assert type(start) is AffineQuantity
    assert type(start.value) is int
    assert start.value == 2**63 + 1
    assert start.unit.delta_unit == Unit('ns')
    elapsed = stop - start
    assert type(elapsed) is Quantity
    assert type(elapsed.value) is int
    assert elapsed == Quantity(3, 'ns')
    assert start + elapsed == stop
    assert elapsed + start == stop
    assert stop - elapsed == start
    assert start.value_in('s') == Fraction(2**63 + 1, 10**9)
    converted = start.to('s')
    assert type(converted) is AffineQuantity
    assert converted == start
    assert converted.value == start.value_in('s')
    assert start < stop
    assert start.value == 2**63 + 1
    with pytest.raises(TypeError, match='points cannot be added'):
        start + stop
    with pytest.raises(TypeError):
        start * 2
    for function in (time.to_timedelta, time.sleep):
        with pytest.raises(TypeError, match='duration Quantity'):
            function(start)


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


@pytest.mark.parametrize(
    'instant',
    [
        datetime.min.replace(tzinfo=timezone.utc),
        datetime.max.replace(tzinfo=timezone.utc),
        datetime(1970, 1, 1, tzinfo=timezone.utc),
        datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc),
        datetime(2026, 10, 22, 12, 34, 56, 123456, tzinfo=timezone(timedelta(hours=8))),
        datetime(2000, 1, 1, tzinfo=timezone(timedelta(seconds=-123, microseconds=456))),
    ],
)
def test_datetime_exact_roundtrip(instant):
    point = time.from_datetime(instant)
    assert type(point) is AffineQuantity
    assert type(point.value) is int
    assert point.unit.delta_unit == Unit('ns')
    utc = time.to_datetime(point)
    assert utc == instant
    assert utc.tzinfo is timezone.utc
    assert time.to_datetime(point, tz=instant.tzinfo) == instant


def test_datetime_epoch_and_native_clock(monkeypatch):
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert time.from_datetime(epoch).value == 0
    assert time.from_datetime(epoch - timedelta(microseconds=1)).value == -1000
    instant = epoch + timedelta(microseconds=1234567)
    monkeypatch.setattr(time._time, 'time_ns', lambda: 1234567000)
    assert time.time() == time.from_datetime(instant)
    assert time.to_datetime(time.time()) == instant


def test_datetime_known_epoch_coordinate():
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    seconds_since_2000 = AffineUnit(time.from_datetime(epoch), delta_unit='s')
    point = time.from_datetime(epoch + timedelta(seconds=2, microseconds=3))
    shifted = point.to(seconds_since_2000)
    assert shifted.value == Fraction(2000003, 1000000)
    assert time.to_datetime(0 @ seconds_since_2000) == epoch
    assert time.to_datetime(shifted) == epoch + timedelta(seconds=2, microseconds=3)
    assert shifted.to('ns') == point
    assert shifted - (0 @ seconds_since_2000) == Quantity(Fraction(2000003, 1000000), 's')


@pytest.mark.parametrize('nanoseconds, microseconds', [(500, 0), (1500, 2), (2500, 2), (-1500, -2)])
def test_datetime_rounding(nanoseconds, microseconds):
    assert time.to_datetime(AffineQuantity(nanoseconds, 'ns')) == datetime(
        1970, 1, 1, tzinfo=timezone.utc
    ) + timedelta(microseconds=microseconds)


def test_datetime_rounding_includes_origin():
    # Rounding the origin and reading separately would produce 0 us, not 1 us.
    coordinate = AffineUnit(Quantity(Fraction(1, 4), 'us'), delta_unit='us')
    assert time.to_datetime(Fraction(1, 4) @ coordinate) == datetime(
        1970, 1, 1, tzinfo=timezone.utc
    )
    assert time.to_datetime(Fraction(1, 2) @ coordinate) == datetime(
        1970, 1, 1, 0, 0, 0, 1, tzinfo=timezone.utc
    )


def test_datetime_decimal_context():
    coordinate = AffineUnit(Quantity(Decimal('0.00000025'), 's'), delta_unit='s')
    with localcontext() as context:
        context.prec = 2
        assert time.to_datetime(Decimal('1.23456725') @ coordinate) == datetime(
            1970, 1, 1, 0, 0, 1, 234568, tzinfo=timezone.utc
        )


def test_datetime_dst_fold():
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        eastern = ZoneInfo('America/New_York')
    except ZoneInfoNotFoundError:
        pytest.skip('IANA timezone data is unavailable')
    first = datetime(2024, 11, 3, 1, 30, tzinfo=eastern, fold=0)
    second = first.replace(fold=1)
    a, b = time.from_datetime(first), time.from_datetime(second)
    assert b - a == Quantity(1, 'h')
    assert time.to_datetime(a, tz=eastern).fold == 0
    assert time.to_datetime(b, tz=eastern).fold == 1


def test_datetime_invalid_input():
    for value in (1, timedelta(0), AffineQuantity(0, 's')):
        with pytest.raises(TypeError, match='datetime'):
            time.from_datetime(value)

    class NoOffset(tzinfo):
        def utcoffset(self, dt):
            return None

    for value in (datetime(2000, 1, 1), datetime(2000, 1, 1, tzinfo=NoOffset())):
        with pytest.raises(ValueError, match='timezone-aware'):
            time.from_datetime(value)
    for value in (1, Quantity(0, 's'), datetime.now(timezone.utc)):
        with pytest.raises(TypeError, match='time point'):
            time.to_datetime(value)
    with pytest.raises(TypeError, match='tzinfo'):
        time.to_datetime(AffineQuantity(0, 's'), tz=None)
    with pytest.raises(DimensionError):
        time.to_datetime(AffineQuantity(1, 'm'))
    logarithmic = AffineUnit(Quantity(1, 's'), delta_unit='dB', kind='power')
    with pytest.raises(DimensionError, match='linear time'):
        time.to_datetime(0 @ logarithmic)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), Decimal('-Infinity')])
def test_datetime_invalid_nonfinite(value):
    with pytest.raises(ValueError, match='finite'):
        time.to_datetime(AffineQuantity(value, 's'))


def test_datetime_numpy_readings_rejected():
    np = pytest.importorskip('numpy')
    for value in (np.float64(0), np.array(0.0), np.array([0.0, 1.0])):
        with pytest.raises(TypeError, match='magnitude'):
            time.to_datetime(AffineQuantity(value, 's'))


def test_datetime_invalid_range():
    with pytest.raises(OverflowError):
        time.from_datetime(datetime.min.replace(tzinfo=timezone(timedelta(hours=1))))
    for instant, offset in (
        (datetime.min.replace(tzinfo=timezone.utc), -1),
        (datetime.max.replace(tzinfo=timezone.utc), 1),
    ):
        with pytest.raises(OverflowError):
            time.to_datetime(time.from_datetime(instant) + Quantity(offset, 'us'))
    with pytest.raises(OverflowError):
        time.to_datetime(
            time.from_datetime(datetime.max.replace(tzinfo=timezone.utc)),
            tz=timezone(timedelta(hours=1)),
        )
