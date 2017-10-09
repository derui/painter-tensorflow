# coding: utf-8


class PerEpochConstantlyUpdater(object):
    def __init__(self, parameter, steps_per_epoch=100, per_epoch=5):

        self._parameter = parameter
        self._step = 0
        self._epoch = 0
        self._per_epoch = per_epoch
        self._steps_per_epoch = steps_per_epoch
        self._epoch_completed = 0

    def __call__(self, *args):
        self._step += 1

        if self._step % self._steps_per_epoch == 0 and self._step > 0:
            self._epoch += 1
            self._step = 0

        if self._epoch >= self._epoch_completed + self._per_epoch:
            self._epoch_completed = self._epoch
            self._parameter.update()


class PerEpochLossUpdater(object):
    def __init__(self, parameter, steps_per_epoch=100, per_epoch=5):

        self._parameter = parameter
        self._step = 0
        self._epoch = 0
        self._per_epoch = per_epoch
        self._steps_per_epoch = steps_per_epoch
        self._epoch_completed = 0

        self._min_loss = -1

    def __call__(self, loss):
        self._step += 1

        if self._step % self._steps_per_epoch == 0 and self._step > 0:
            self._epoch += 1
            self._step = 0

        if self._min_loss < 0 or self._min_loss > loss:
            self._min_loss = loss
            self._epoch_completed = self._epoch
        else:
            if self._epoch >= self._epoch_completed + self._per_epoch:
                self._epoch_completed = self._epoch
                self._min_loss = loss
                self._parameter.update()


class UpdatableParameter(object):
    def __init__(self, initial, update_rate, dtype=None):

        self._update_rate = update_rate
        self._value = initial

    def __call__(self):
        return self._value

    def update(self):
        self._value *= self._update_rate
