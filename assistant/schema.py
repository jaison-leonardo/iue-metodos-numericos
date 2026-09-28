"""
assistant/schema.py
=====================
Modelos de datos (contrato) entre la interpretación del ejercicio y el
resto del pipeline determinista, según la sección 6 de la
especificación. Se implementan como `dataclasses` estándar de Python
(sin agregar una dependencia nueva como pydantic todavía); si más
adelante se integra el parser LLM con validación más estricta de JSON
arbitrario, se puede migrar a pydantic sin cambiar el resto del
pipeline (validator.py / executor.py consumen estos objetos, no JSON
crudo).

Alcance de esta primera versión (aprobado): solo
    - root_finding   (bisección, falsa posición, Newton, secante)
    - pvi_first_order (Euler, Heun, Punto Medio, RK4)
PVI de 2do orden y PVF quedan fuera del asistente por ahora (el código
existente de PVI_2Ord/ y pvf/ no se toca ni se integra todavía).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


ROOT_METHODS = ("biseccion", "falsapos", "newton", "secante", "puntofijo")
PVI1_METHODS = ("euler", "heun", "ptomed", "rk4")


@dataclass
class RootFindingProblem:
    """Especificación estructurada de un problema de búsqueda de raíces."""
    fx: str                         # expresión de f(x), p.ej. "x**3 + 3*x - 32"
    method: str                     # uno de ROOT_METHODS
    a0: Optional[float] = None      # extremo izquierdo (métodos cerrados)
    b0: Optional[float] = None      # extremo derecho (métodos cerrados)
    x0: Optional[float] = None      # aproximación inicial (Newton / secante / punto fijo)
    x1: Optional[float] = None      # segunda aproximación inicial (secante)
    gx: Optional[str] = None        # g(x) tal que x = g(x), requerido solo por 'puntofijo'
    tol: float = 1e-6
    maxit: int = 50
    rn: List[int] = field(default_factory=lambda: [0])
    display_precision: int = 4
    exercise_text: Optional[str] = None  # enunciado original, para trazabilidad


@dataclass
class PVIProblem:
    """Especificación estructurada de un PVI de primer orden y' = f(t,y)."""
    ode_rhs: str                    # f(t, y), p.ej. "-k*y"
    method: str                     # uno de PVI1_METHODS
    t0: float
    y0: float
    t_end: float
    h: float
    parameters: dict = field(default_factory=dict)  # p.ej. {"k": 0.175}
    analytical_solution: Optional[str] = None        # y(t) exacta, si el enunciado la da
    evento_y: Optional[float] = None  # valor de "y" en el que detener la
        # integracion ANTES de llegar a t_end (parada por evento, ej.
        # y=0 para "tiempo que tarda en vaciarse el tanque"). t_end
        # sigue siendo obligatorio: actua como cota superior de la
        # busqueda por si el evento nunca se alcanza. Si es None
        # (default), el comportamiento es identico al de siempre:
        # se integra hasta t_end sin ninguna condicion de parada extra.
    rn: List[int] = field(default_factory=lambda: [0])
    display_precision: int = 4
    exercise_text: Optional[str] = None


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)

    def raise_if_invalid(self):
        if not self.ok:
            raise ValueError("Problema inválido: " + "; ".join(self.errors))


@dataclass
class NeedsClarification:
    """
    Corresponde a la sección 21 de la especificación: cuando el
    ejercicio es ambiguo, el sistema NO debe adivinar, debe devolver
    esto en lugar de una especificación estructurada.
    """
    status: str = "needs_clarification"
    reason: str = ""
    missing_information: List[str] = field(default_factory=list)


@dataclass
class RootFindingResult:
    problem: RootFindingProblem
    r: float
    iterac: int
    error_residual: float
    trace: List[dict]              # traza COMPLETA (todas las iteraciones)


@dataclass
class PVIResult:
    problem: PVIProblem
    t: list
    y: list                        # arreglo completo (índice 0 = condición inicial)
    evento_alcanzado: Optional[bool] = None  # True si se detuvo por evento_y
    t_evento: Optional[float] = None         # instante interpolado del cruce
    paso_evento: Optional[int] = None        # indice (en t/y) del cruce
    evento_aproximado: Optional[bool] = None  # True si t_evento vino del
        # fallback de extrapolacion lineal (el paso completo se salio del
        # dominio, ej. sqrt(y) con y<0 por sobrepaso numerico cerca de 0)
        # en vez de interpolacion exacta entre dos puntos finitos.


@dataclass
class ComparisonResult:
    """Salida del comparador (sección 19/20 de la especificación)."""
    numeric_value: float
    reference_value: Optional[float]
    reference_source: Optional[str]   # "analytical" | "solve_ivp" | "residual"
    absolute_error: Optional[float]
    relative_error: Optional[float]
    converged: bool
