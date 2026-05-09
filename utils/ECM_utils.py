import numpy as np
import re
import matplotlib.pyplot as plt

class CircuitParams:
    """
    Generic circuit parser for impedance frequency response.
    """

    def __init__(self, circuit):

        '''
        :param circuit: a string representing a circuit expression
        '''

        if circuit is None or (isinstance(circuit, (str, tuple)) and len(circuit) == 0):
            raise ValueError('[EquivalentCircuit] "circuit" is empty!')
        self.circuit = circuit.replace(' ', '')

        # counters for automatic naming
        self.counts = {
            "R": 0,
            "C": 0,
            "L": 0,
            "CPE": 0,
            "W": 0
        }
        self.pos = 0
        self.param_names = []

        pattern = r'CPE|R|C|L|W|\(|\)|\+|//'
        self.tokens = re.findall(pattern, self.circuit)
        self.tree = self.circuit_tree()

        if self.pos != len(self.tokens):
            raise ValueError("Invalid circuit: extra tokens found")

    def circuit_tree(self):
        values = []  # elements and partial trees
        ops = []  # '+', '//', '('

        precedence = {
            '+': 1,
            '//': 2
        }

        for token in self.tokens:
            if self.pos >= len(self.tokens):
                raise ValueError("Unexpected end of circuit")

            token = self.tokens[self.pos]

            if token == '(':
                ops.append('(')
                self.pos += 1

            elif token in ("R", "C", "L", "CPE", "W"):
                values.append(self.make_element(token))
                self.pos += 1

            elif token in ('+', '//'):
                while (
                        ops
                        and ops[-1] != '('
                        and precedence[ops[-1]] >= precedence[token]
                ):
                    if len(values) < 2:
                        raise ValueError("Invalid circuit: missing operand")

                    right = values.pop()
                    left = values.pop()
                    op = ops.pop()

                    if op == '+':
                        values.append(("series", left, right))
                    elif op == '//':
                        values.append(("parallel", left, right))

                ops.append(token)
                self.pos += 1

            elif token == ')':
                while ops and ops[-1] != '(':
                    if len(values) < 2:
                        raise ValueError("Invalid circuit: missing operand")

                    right = values.pop()
                    left = values.pop()
                    op = ops.pop()

                    if op == '+':
                        values.append(("series", left, right))
                    elif op == '//':
                        values.append(("parallel", left, right))

                if not ops or ops[-1] != '(':
                    raise ValueError("Missing opening parenthesis")

                ops.pop()  # remove '('
                self.pos += 1

            else:
                raise ValueError(f"Unexpected token: {token}")

        while ops:
            if ops[-1] == '(':
                raise ValueError("Missing closing parenthesis")

            if len(values) < 2:
                raise ValueError("Invalid circuit: missing operand")

            right = values.pop()
            left = values.pop()
            op = ops.pop()

            if op == '+':
                values.append(("series", left, right))
            elif op == '//':
                values.append(("parallel", left, right))

        if len(values) != 1:
            raise ValueError("Invalid circuit expression")

        self.tree = values[0]
        return self.tree

    def make_element(self, kind):

        self.counts[kind] += 1
        n = self.counts[kind]

        if kind == "R":
            name = f"R{n}"
            self.param_names.append(name)
            return ("R", name)

        elif kind == "C":
            name = f"C{n}"
            self.param_names.append(name)
            return ("C", name)

        elif kind == "L":
            name = f"L{n}"
            self.param_names.append(name)
            return ("L", name)

        elif kind == "CPE":
            q_name = f"Q{n}"
            alpha_name = f"alpha{n}"
            self.param_names.extend([q_name, alpha_name])
            return ("CPE", q_name, alpha_name)

        elif kind == "W":
            name = f"W{n}"
            self.param_names.append(name)
            return ("W", name)


class CircuitEvaluate:
    def __init__(self, freqs = np.array, params = np.array, tree = None):
     '''

     :param freqs: list of frequencies
     :param params: list of initial parameters
     :param tree: circuit tree
     '''


     self.Freqs = np.asarray(freqs, dtype=float)
     self.W = 2 * np.pi * self.Freqs
     self.params = params
     self.tree = tree
     self.Z_ECM = self.eval_node(self.tree)

     Z_ECM_real = np.real(self.Z_ECM)
     Z_ECM_imag = np.imag(self.Z_ECM)
     Z_ECM_mag = np.abs(self.Z_ECM)
     Z_ECM_phase = np.angle(self.Z_ECM, deg=True)

     fig, ax = plt.subplots(2, 1, figsize=(8, 5))
     fig.suptitle("ECM - Frequency Evaluation")
     ax1, ax2 = ax
     # --- Bode magnitude ---
     ax1.semilogx(freqs, Z_ECM_mag)
     ax1.set_xlabel("Frequency (Hz)")
     ax1.set_ylabel("|Z| (Ohm)")
     ax1.set_title("Bode Magnitude")
     ax1.grid(True, which="both")
     # --- Bode phase ---
     ax2.semilogx(freqs, Z_ECM_phase)
     ax2.set_xlabel("Frequency (Hz)")
     ax2.set_ylabel("Phase (deg)")
     ax2.set_title("Bode Phase")
     ax2.grid(True, which="both")
     plt.tight_layout()
     plt.show()

     # Nyquist
     plt.figure(figsize=(6, 6))
     plt.suptitle("ECM - Frequency Evaluation")
     plt.plot(Z_ECM_real, -Z_ECM_imag)
     plt.xlabel("Z' (Ohm)")
     plt.ylabel("-Z'' (Ohm)")
     plt.title("Nyquist Plot")
     plt.grid(True)
     plt.tight_layout()
     plt.show()

    def eval_node(self, node):
        """
         Recursively evaluate one node of the circuit tree.
         """
        kind = node[0]

        if kind == "series":
            z_left = self.eval_node(node[1])
            z_right = self.eval_node(node[2])
            return self.z_series(z_left, z_right)

        elif kind == "parallel":
            z_left = self.eval_node(node[1])
            z_right = self.eval_node(node[2])
            return self.z_parallel(z_left, z_right)

        elif kind == "R":
            return self.z_R(self.params[node[1]], self.W)

        elif kind == "C":
            return self.z_C(self.params[node[1]], self.W)

        elif kind == "L":
            return self.z_L(self.params[node[1]], self.W)

        elif kind == "CPE":
            Q = self.params[node[1]]
            alpha = self.params[node[2]]
            return self.z_CPE(Q, alpha, self.W)

        elif kind == "W":
            return self.z_W(self.params[node[1]], self.W)

        else:
            raise ValueError(f"Unknown node type: {kind}")


    def z_series(self, z1, z2):
        return z1 + z2

    def z_parallel(self, z1, z2):
        return 1 / (1 / z1 + 1 / z2)

    def z_R(self, R, w):
        return R * np.ones_like(w, dtype=complex)

    def z_C(self, C, w):
        return 1 / (1j * w * C)

    def z_L(self, L, w):
        return 1j * w * L

    def z_CPE(self, Q, alpha, w):
        return 1 / (Q * (1j * w) ** alpha)

    def z_W(self, sigma, w):
        """
         Semi-infinite Warburg:
             Z = sigma / sqrt(j*w)
         """
        return sigma / np.sqrt(1j * w)
