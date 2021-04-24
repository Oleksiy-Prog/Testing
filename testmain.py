import unittest
import main
from numpy import array_equal


class TestCountRisk(unittest.TestCase):
    """ Тестирование подсчета матрицы рисков """

    def test_count_risk_matrix_1(self):
        MatrixStart = [
            [1, 4, 5, 9],
            [3, 8, 4, 3],
            [4, 6, 6, 2]
        ]

        MatrixRisk = [
            [3, 4, 1, 0],
            [1, 0, 2, 6],
            [0, 2, 0, 7]
        ]

        MatrixResult = main.count_risk(MatrixStart)

        assert array_equal(MatrixResult, MatrixRisk) == True

    def test_count_risk_matrix_2(self):
        MatrixStart = [
            [5, 3, 4, 2, 1],
            [5, 3, 2, 1, 1],
            [1, 2, 5, 4, 3],
            [7, 6, 7, 3, 1],
            [1, 2, 3, 4, 3]
        ]

        MatrixRisk = [
            [2, 3, 3, 2, 2],
            [2, 3, 5, 3, 2],
            [6, 4, 2, 0, 0],
            [0, 0, 0, 1, 2],
            [6, 4, 4, 0, 0]
        ]

        MatrixResult = main.count_risk(MatrixStart)

        assert array_equal(MatrixResult, MatrixRisk) == True


class TestProbability(unittest.TestCase):
    """ Тестирование решения, полученного из критерия, основанного на известных вероятностях условиях """

    def test_with_probaility_1(self):
        MatrixRisk = [
            [3, 4, 1, 0],
            [1, 0, 2, 6],
            [0, 2, 0, 7]
        ]

        right_result_list = [2.5, 1.4, 1.5]

        probability_list = [0.2, 0.4, 0.3, 0.1]

        result_list = main.with_probability(MatrixRisk, probability_list)

        assert array_equal(result_list, right_result_list) == True

    def test_with_probaility_2(self):
        MatrixRisk = [
            [2, 3, 3, 2, 2],
            [2, 3, 5, 3, 2],
            [6, 4, 2, 0, 0],
            [0, 0, 0, 1, 2],
            [6, 4, 4, 0, 0]
        ]

        right_result_list = [2.4, 3.0, 2.4, 0.6, 2.8]

        probability_list = [0.2, 0.2, 0.2, 0.2, 0.2]

        result_list = main.with_probability(MatrixRisk, probability_list)

        assert array_equal(result_list, right_result_list) == True


class TestWald(unittest.TestCase):
    """ Тестирование решения, полученного из критерия Вальда """

    def test_Wald_1(self):
        MatrixPay = [
            [1, 4, 5, 9],
            [3, 8, 4, 3],
            [4, 6, 6, 2]
        ]

        right_result_list = [1, 3, 2]

        result_list = main.Wald(MatrixPay)

        assert array_equal(result_list, right_result_list) == True

    def test_Wald_2(self):
        MatrixPay = [
            [5, 3, 4, 2, 1],
            [5, 3, 2, 1, 1],
            [1, 2, 5, 4, 3],
            [7, 6, 7, 3, 1],
            [1, 2, 3, 4, 3]
        ]

        right_result_list = [1, 1, 1, 1, 1]

        result_list = main.Wald(MatrixPay)

        assert array_equal(result_list, right_result_list) == True


class TestSavage(unittest.TestCase):
    """ Тестирование решения, полученного из критерия Сэвиджа """

    def test_Savage_1(self):
        MatrixRisk = [
            [3, 4, 1, 0],
            [1, 0, 2, 6],
            [0, 2, 0, 7]
        ]

        right_result_list = [4, 6, 7]

        result_list = main.Savage(MatrixRisk)

        assert array_equal(result_list, right_result_list) == True

    def test_Savage_2(self):
        MatrixRisk = [
            [2, 3, 3, 2, 2],
            [2, 3, 5, 3, 2],
            [6, 4, 2, 0, 0],
            [0, 0, 0, 1, 2],
            [6, 4, 4, 0, 0]
        ]

        right_result_list = [3, 5, 6, 2, 6]

        result_list = main.Savage(MatrixRisk)

        assert array_equal(result_list, right_result_list) == True


class TestHurwitzMatrix(unittest.TestCase):
    """ Тестирование решения, полученного из критерия Гурвица, основанном на выигрыше """

    def test_Hurwitz_matrix_1(self):
        MatrixPay = [
            [1, 4, 5, 9],
            [3, 8, 4, 3],
            [4, 6, 6, 2]
        ]

        right_result_list = [5.0, 5.5, 4.0]

        result_list = main.Hurwitz_matrix(MatrixPay)

        assert array_equal(result_list, right_result_list) == True

    def test_Hurwitz_matrix_2(self):
        MatrixPay = [
            [5, 3, 4, 2, 1],
            [5, 3, 2, 1, 1],
            [1, 2, 5, 4, 3],
            [7, 6, 7, 3, 1],
            [1, 2, 3, 4, 3]
        ]

        right_result_list = [3.0, 3.0, 3.0, 4.0, 2.5]

        result_list = main.Hurwitz_matrix(MatrixPay)

        assert array_equal(result_list, right_result_list) == True


class TestHurwitzRisk(unittest.TestCase):
    """ Тестирование решения, полученного из критерия Гурвица, основанном на риске """

    def test_Hurwitz_risk_1(self):
        MatrixRisk = [
            [3, 4, 1, 0],
            [1, 0, 2, 6],
            [0, 2, 0, 7]
        ]

        right_result_list = [2.0, 3.0, 3.5]

        result_list = main.Hurwitz_risk(MatrixRisk)

        assert array_equal(result_list, right_result_list) == True

    def test_Hurwitz_risk_2(self):
        MatrixRisk = [
            [2, 3, 3, 2, 2],
            [2, 3, 5, 3, 2],
            [6, 4, 2, 0, 0],
            [0, 0, 0, 1, 2],
            [6, 4, 4, 0, 0]
        ]

        right_result_list = [2.5, 3.5, 3.0, 1.0, 3.0]

        result_list = main.Hurwitz_risk(MatrixRisk)

        assert array_equal(result_list, right_result_list) == True


if __name__ == "__main__":
    unittest.main()
