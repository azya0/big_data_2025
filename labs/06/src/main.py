import part1 as p1
import part2 as p2

if __name__ == "__main__":
    data = p1.generate_data()
    lin_res = p1.build_linear_regression(data)
    p1.plot_linear_regression(data, lin_res)
    p1.calc_and_print_results(data, lin_res)

    m_data = p2.modify_data(data)
    mul_res = p2.build_multiplicative_model(m_data)
    p2.plot_multiplicative_model(m_data, mul_res)
    p2.calc_and_print_results(m_data, mul_res)
