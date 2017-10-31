import numpy
import glob
import pathlib
import collections

from .scalar_stats import moving_mean_std
from .scalar_stats import regress
from .image import colorize
from . import datafile

def read_worms(*path_globs, name_prefix='', delimiter='\t', last_timepoint_is_first_dead=True, age_scale=1):
    """Read worm data files into a Worms object.

    Each file can be either a single-worm file (where the "timepoint" identifier
    is the first column), or a multi-worm file (which has timepoint data for
    multiple worms concatenated, and has "name" as the column to disambiguate
    each worm).

    In the case of a single-worm file, the name of the worm is the name of the
    input file (minus any file extension), with an optional prefix prepended.
    In the case of a multi-worm file, the name of the worm is given in the
    "name" column, again optionally prepended by a prefix.

    The prefix is useful to distinguish animals from different experimental runs
    or genotypes, &c.

    Each data file must have at a minimum a "timepoint" column (assumed to be
    the first column) and an "age" column.

    Parameters:
        *path_globs: all non-keyword arguments are treated as paths to worm data
            files, or as glob expressions matching multiple such data files.
        name_prefix: if a string, this will be used as the prefix (see above).
            If a function, it will be called with the pathlib.Path of the input
            file in question, so that the file name and/or name of the parent
            directories can be used to compute the prefix.
        delimiter: controls whether the data are assumed to be tab, comma, or
            otherwise delimited.
        last_timepoint_is_first_dead: if True, the last timepoint in the data
            file is assumed to represent the first time the worm was annotated
            as dead. Otherwise, the last timepoint is assumed to represent the
            last time the worm was known to be alive.
        age_scale: scale-factor for the ages read in. The values in the "age"
            column and any column ending in '_age' will be multiplied by this
            scalar. (Useful e.g. for converting hours to days.)

    Returns: Worms object

    Examples:
        worms = read_worms('/path/to/spe-9/datafiles/*.csv', name_prefix='spe-9 ', delimiter=',')

        def get_genotype(path):
            # assume that the grandparent path contains the genotype
            return path.parent.parent.name + ' '
        worms = read_worms('/path/to/*/datafiles/*.tsv', name_prefix=get_genotype)
    """
    worms = Worms()
    for path_glob in path_globs:
        for path in map(pathlib.Path, glob.iglob(path_glob, recursive=True)):
            if callable(name_prefix):
                prefix = name_prefix(path)
            else:
                prefix = name_prefix
            for name, header, data in _read_datafile(path, prefix, delimiter):
                worms.append(Worm(name, header, data, last_timepoint_is_first_dead, age_scale))
    worms.sort('lifespan')
    return worms

def _read_datafile(path, prefix, delimiter):
    """Iterate over a single- or multi-worm datafile, yielding (name, header, data)
    triplets corresponding to each worm in the file."""
    header, data = datafile.read_delimited(path, delimiter=delimiter)
    is_multi_worm = header[0] in ('name', 'worm')
    if not is_multi_worm:
        name = prefix + path.stem
        header[0] = 'timepoint'
        yield name, header, data
    else:
        header[1] = 'timepoint'
        worm_rows = []
        current_name = None
        for name, *row in data:
            if current_name is None:
                current_name = name
            if current_name != name:
                yield prefix + current_name, header[1:], worm_rows
                current_name = name
                worm_rows = []
            worm_rows.append(row)
        yield prefix + current_name, header[1:], worm_rows

def meta_worms(grouped_worms, *timecourse_features, age_feature='age', summary_features=('lifespan',), smooth=0.4):
    """Calculate average trends across groups of worms, returning a new Worms object.

    Given a set of timecourse features and a set of worms grouped by some criterion,
    calculate the average trajectory of each given feature across all group members,
    using LOWESS smoothing (as implemented in zplib.scalar_stats.moving_mean_std.moving_mean)

    Each produced meta "worm" will have the averaged timecourse features, the
    "name" of the group, the average of any specified summary features (such as
    "lifespan"), and attributes "worms" and "n" that list the underlying worms
    and the count of the same.

    Parameters:
        grouped_worms: dictionary mapping from group names to Worms objects of
            the worms in each group. The group name will be the "name" attribute
            of each "meta worm" produced. Such a grouped_worms dict can be
            produced from the Worms.bin or Worms.group_by_value functions.
        *timecourse_features: one or more feature names, listing the features
            for which time averages should be calculated.
        age_feature: feature to use for the "age" axis of the average trajectories.
            Generally "age" is right, but trajectories could be centered on
            time of death using a "ghost_age" feature that counts down to zero
            at the time of death. This parameter may be a callable function or
            a feature name.
        summary_features: names of summary feature to average across each of the
            underlying worms.
        smooth: smoothing parameter passed to LOWESS moving_mean function.
            Represents the fraction of input data points that will be smoothed
            across at each output data point.

    Returns:
        Worms object, sorted by group name.

    Example:
    lifespan_bins = worms.bin('lifespan', nbins=5)
    averaged_worms = meta_worms(lifespan_bins, 'gfp_95th')
    averaged_worms.plot_timecourse('gfp_95th')

    """
    meta_worms = Worms()
    for group_name, worms in sorted(grouped_worms.items()):
        meta_worm = Worm(group_name)
        meta_worm.n = len(worms)
        meta_worm.worms = worms
        ages = numpy.concatenate([age_feature(worm) if callable(age_feature) else getattr(worm.td, age_feature) for worm in worms])
        for feature in timecourse_features:
            vals = numpy.concatenate([getattr(worm.td, feature) for worm in worms])
            ages_out, trend = moving_mean_std.moving_mean(ages, vals, points_out=50, smooth=smooth, iters=1)
            setattr(meta_worm.td, feature, trend)
        for feature in summary_features:
            setattr(meta_worm, feature, worms.get_feature(feature).mean())
        meta_worm.td.age = ages_out # the same for all loops; just set once
        meta_worms.append(meta_worm)
    return meta_worms

class Worm(object):
    """Object for storing data pertaining to an indiviual worm's life.

    The object has an attribute, 'worm.td' (for "timecourse data") to contain
    measurements made at each timepoint. These include at a minimum "timepoint"
    (the string identifier for each timepoint, generally a timestamp) and "age".

    Other attributes, such as worm.lifespan and worm.name represent data or
    summary statistics valid over the worm's entire life.

    Convenience accessor functions for getting a range of timecourse measurements
    are provided.
    """
    def __init__(self, name, feature_names=[], timepoint_data=[], last_timepoint_is_first_dead=True, age_scale=1):
        """It is generally preferable to construct worms from a factory function
        such as read_worms or meta_worms, rather than using the constructor.

        Parameters:
            name: identifier for this individual animal
            feature_names: names of the timecourse features measured for
                this animal
            timepoint_data: for each timepoint, each of the measured features.
            last_timepoint_is_first_dead: if True, the last timepoint in the
                data file is assumed to represent the first time the worm was
                annotated as dead. Otherwise, the last timepoint is assumed to
                represent the last time the worm was known to be alive.
            age_scale: scale-factor for the ages read in. The values in the
                "age" column and any column ending in '_age' will be multiplied
                by this scalar. (Useful e.g. for converting hours to days.)
        """
        self.name = name
        self.td = _TimecourseData()
        vals_for_features = [[] for _ in feature_names]
        for timepoint in timepoint_data:
            # add each entry in the timepoint data to the corresponding list in
            # vals_for_features
            for feature_vals, item in zip(vals_for_features, timepoint):
                feature_vals.append(item)
        for feature_name, feature_vals in zip(feature_names, vals_for_features):
            arr = numpy.array(feature_vals)
            if feature == 'age' or feature.endswith('_age'):
                arr *= age_scale
            setattr(self.td, feature.replace(' ', '_'), arr)
        if hasattr(self.td, 'age'):
            if last_timepoint_is_first_dead:
                self.lifespan = self.td.age[-2:].mean() # midpoint between last-alive and first-dead
            else:
                self.lifespan = self.td.age[-1] + (self.td.age[-1] - self.td.age[-2])/2 # halfway through the next interval, assumes equal intervals

    def __repr__(self):
        return 'Worm("{}")'.format(self.name)

    def get_time_range(self, feature, age_min=-numpy.inf, age_max=numpy.inf, age_feature='age', match_closest=False):
        """

        """
        ages = age_feature(self) if callable(age_feature) else getattr(self.td, age_feature)
        data = feature(self) if callable(feature) else getattr(self.td, feature)
        if match_closest:
            mask = self._get_closest_times_mask(ages, age_min, age_max)
        else:
            mask = (ages >= age_min) & (ages <= age_max)
        if numpy.issubdtype(data.dtype, float):
            mask &= ~numpy.isnan(data)
        return ages[mask], data[mask]

    @staticmethod
    def _get_closest_times_mask(ages, age_min, age_max):
        li = numpy.argmin(numpy.absolute(ages - age_min))
        ui = numpy.argmin(numpy.absolute(ages - age_max))
        r = numpy.arange(len(ages))
        return (r >= li) & (r <= ui)

    def interp_feature(self, feature, age):
        ages, value = self.get_time_range(feature)
        return numpy.interp(age, ages, value)

class Worms(collections.UserList):
    def read_summary_data(self, path, ignore_new=True, delimiter='\t'):
        named_worms = {worm.name: worm for worm in self}
        header, data = datafile.read_delimited(path, delimiter=delimiter)
        for name, *rest in data:
            if name not in named_worms:
                if ignore_new:
                    print('Record read for unknown worm "{}"'.format(name))
                    continue
                else:
                    worm = Worm(name)
                    self.append(worm)
            else:
                worm = named_worms[name]
            for feature, val in zip(header[1:], rest):
                setattr(worm, feature, val)

    def write_summary_data(self, path, features=None, delimiter='\t', error_on_missing=True):
        if features is None:
            features = set()
            for worm in self:
                features.update(worm.__dict__.keys())
            features.remove('name')
            features.remove('td')
            features = ['name'] + sorted(features)
        data = [features]
        for worm in self:
            row = []
            for feature in features:
                if feature not in worm.__dict__ and error_on_missing:
                    raise ValueError('Worm "{worm.name}" has no "{feature}" feature.')
                else:
                    row.append(getattr(worm, feature, ''))
            data.append(row)
        datafile.write_delimited(path, data, delimiter)

    def write_timecourse_data(self, path, multi_worm_file=False, features=None, suffix=None, delimiter='\t', error_on_missing=True):
        path = pathlib.Path(path)
        if not multi_worm_file:
            path.mkdir(parents=False, exist_ok=False)
            if suffix is None:
                if delimiter == ',':
                    suffix = '.csv'
                elif delimiter == '\t':
                    suffix = '.tsv'
                else:
                    suffix = '.dat'
        if features is None:
            features = set()
            for worm in self:
                features.update(worm.td.__dict__.keys())
            features.remove('timepoint')

            features = ['timepoint'] + sorted(features)
        if multi_worm_file:
            data = [['name'] + features]
        for worm in self:
            n = len(worm.td.timepoint)
            missing = [''] * n
            cols = []
            for feature in features:
                if feature not in worm.__dict__ and error_on_missing:
                    raise ValueError('Worm "{worm.name}" has no "{feature}" feature.')
                else:
                    cols.append(getattr(worm.td, feature, missing))
            assert all(len(c) == n for c in cols)
            rows = [[] for _ in range(n)]
            for col in cols:
                for row, colval in zip(rows, col):
                    row.append(colval)
            if multi_worm_file:
                for row in rows:
                    row.insert(worm.name, 0)
                data.extend(rows)
            else:
                worm_path = (path / worm.name).with_suffix(suffix)
                datafile.write_delimited(worm_path, [features] + rows, delimiter)
        if multi_worm_file:
            datafile.write_delimited(path, data, delimiter)

    def merge_in(self, input_worms):
        named_worms = {worm.name: worm for worm in self}
        for input_worm in input_worms:
            if input_worm.name not in named_worms:
                print("no match found for", input_worm.name)
                continue
            our_worm = named_worms[input_worm.name]
            # merge timecourse data
            for k, v in input_worm.td.__dict__.items():
                if hasattr(our_worm.td, k):
                    # if both worms have the same named timecourse information, make sure it matches
                    assert numpy.all(v == getattr(our_worm.td, k))
                else:
                    setattr(our_worm.td, k, v)
            # merge summary data
            for k, v in input_worm.__dict__.items():
                if k == 'td':
                    pass
                if hasattr(our_worm, k):
                    assert numpy.all(v == getattr(our_worm, k))
                else:
                    setattr(our_worm, k, v)

    def sort(self, feature, reverse=False):
        if callable(feature):
            key = feature
        else:
            def key(worm):
                getattr(worm, feature)
        super().sort(key=key, reverse=reverse)

    def filter(self, criterion):
        mask = self.get_feature(criterion).astype(bool)
        return self.__class__([worm for worm, keep in zip(self, mask) if keep])

    def get_time_range(self, feature, min_age=-numpy.inf, max_age=numpy.inf, age_feature='age'):
        out = []
        for worm in self:
            ages, value = worm.get_time_range(feature, min_age, max_age, age_feature)
            out.append(numpy.transpose([ages, value]))
        return out

    def get_feature(self, feature):
        return numpy.array([feature(worm) if callable(feature) else getattr(worm, feature) for worm in self])

    def get_features(self, *features):
        vals = list(map(self.get_feature, features))
        if len(features) > 1:
            vals = numpy.concatenate(vals, axis=1)
        else:
            vals = vals[0]
        if vals.ndim == 1:
            vals = vals[:, numpy.newaxis]
        return vals

    def group_by_value(self, values):
        assert len(values) == len(self)
        worms = collections.defaultdict(self.__class__)
        for worm, value in zip(self, values):
            worms[value].append(worm)
        return dict(worms)

    def bin(self, feature, nbins, equal_count=False):
        data = self.get_feature(feature)
        if equal_count:
            ranks = data.argsort().argsort()
            bins = list(map(str, (nbins * ranks / (len(ranks))).astype(int)))
        else:
            bin_edges = numpy.linspace(data.min(), data.max(), nbins+1)
            bin_numbers = numpy.digitize(data, bin_edges[1:-1])
            bin_names = ['[{:.1f}-{:.1f})'.format(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-2)]
            bin_names.append('[{:.1f}-{:.1f}]'.format(bin_edges[-2], bin_edges[-1]))
            bins = [bin_names[n] for n in bin_numbers]
        return self.group_by_value(bins)

    def regress(self, *features, target='lifespan', control_features=None, regressor=None, filter_valid=True):
        y = self.get_feature(target)
        X = self.get_features(*features)
        C = None if control_features is None else self.get_features(*control_features)
        if filter_valid:
            valid_mask = numpy.isfinite(X).all(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            if C is not None:
                C = C[valid_mask]
        return regress.regress(X, y, C, regressor)

    def regress_time_data(self, *features, target='age', min_age=-numpy.inf, max_age=numpy.inf, age_feature='age', regressor=None):
        X = []
        for feature in features:
            X.append(numpy.concatenate(self.get_time_range(feature, min_age, max_age, age_feature), axis=1)[:,1])
        X = numpy.transpose(X)
        y = numpy.concatenate(self.get_time_range(target, min_age, max_age, age_feature), axis=1)[:,1]
        return _regress(X, y, regressor)

    def _timecourse_plot_data(self, feature, min_age=-numpy.inf, max_age=numpy.inf, age_feature='age', color_by='lifespan'):
        time_ranges = self.get_time_range(feature, min_age, max_age, age_feature)
        color_vals = colorize.scale(self.get_feature(color_by), output_max=1)
        colors = colorize.color_map(color_vals, uint8=False)
        out = []
        for time_range, color in zip(time_ranges, colors):
            x, y = time_range.T
            out.append((x, y, color))
        return out

    def plot_timecourse(self, feature, min_age=-numpy.inf, max_age=numpy.inf, age_feature='age'):
        import matplotlib.pyplot as plt
        plt.clf()
        for x, y, c in self._timecourse_plot_data(feature, min_age, max_age, age_feature):
            plt.plot(x, y, color=c)

class _TimecourseData(object):
    def __repr__(self):
        return 'Timecourse Data:\n' + '\n'.join('    ' + k for k in sorted(self.__dict__) if not k.startswith('_'))

