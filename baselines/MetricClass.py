import abc


class MetricClass(metaclass=abc.ABCMeta):
    '''
    This class is an abstract class for metrics
    '''

    @abc.abstractmethod
    def __call__(self, ref, hyp):
        '''
        This function calculates a metric given all of its parameters (ref could also be src)
        :return: score
        '''
        raise NotImplementedError


    def evaluate_df(self, df):
        return self.__call__(df['SRC'].tolist(), df['HYP'].tolist())