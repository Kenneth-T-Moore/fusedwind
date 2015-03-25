
import numpy as np
from math import factorial
from scipy.optimize import leastsq

from openmdao.lib.datatypes.api import List, Bool, Array, Instance, Int, Float, VarTree
from openmdao.main.api import Component

from fusedwind.turbine.geometry_vt import Curve, AirfoilShape
from fusedwind.lib.distfunc import distfunc
from fusedwind.lib.geom_tools import calculate_length
from fusedwind.lib.cubicspline import NaturalCubicSpline


def _C(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))


class BezierCurve(Curve):
    """
    Computes a 2D/3D bezier curve
    """

    CPs = Array(desc='Array of control points')

    def add_control_point(self, p):

        C = list(self.CPs)
        C.append(list(p))
        self.CPs = np.asarray(C)

    def update(self):

        try:
            self.nd = self.CPs.shape[1]
        except:
            raise RuntimeError('CPs needs to an array of shape (m, n)')

        if self.ni == 0:
            self.ni = 100

        points = self._compute(self.CPs)
        self._s = calculate_length(points)
        self._s /= self._s[-1]
        self.initialize(points)

    # def _compute_dp(self):
    #     """
    #     computes the derivatives (tangent vectors) along a Bezier curve
    #     wrt ``t``.

    #     there is no trivial analytic function to compute derivatives wrt
    #     to a given space interval, so we just spline and redistribute
    #     see: http://pomax.github.io/bezierinfo/
    #     """
    #     C = np.zeros((self.CPs.shape[0] - 1, self.nd))
    #     nC = C.shape[0]
    #     for i in range(nC):
    #         C[i, :] = float(nC) * (self.CPs[i + 1] - self.CPs[i])

    #     dp = self._compute(C)

    def _compute(self, C):

        points = np.zeros((self.ni, self.nd), dtype=C.dtype)
        self.t = np.linspace(0., 1., self.ni, dtype=C.dtype)
        # control point iterator
        _n = xrange(C.shape[0])

        for i in range(self.ni):
            s = self.t[i]
            n = _n[-1]
            for j in range(self.nd):
                for m in _n:
                    # compute bernstein polynomial
                    b_i = _C(n, m) * s**m * (1 - s)**(n - m)
                    # multiply ith control point by ith bernstein polynomial
                    points[i, j] += C[m, j] * b_i

        return points


class BezierAirfoilShape(Component):

    ni = Int(200, iotype='in')
    fix_x = Bool(True, iotype='in')
    symmLE = Bool(True, iotype='in')
    symmTE = Bool(True, iotype='in')

    afIn = VarTree(AirfoilShape(), iotype='in')
    spline_CPs = Array(iotype='in', units=None, desc='Spline starting control points')

    CPu = Array(iotype='in', fd_form='complex_step', fd_step=1.e-20)
    CPl = Array(iotype='in', fd_form='complex_step', fd_step=1.e-20)
    CPle = Float(iotype='in', fd_form='complex_step', fd_step=1.e-20)
    CPte = Float(iotype='in', fd_form='complex_step', fd_step=1.e-20)

    afOut = VarTree(AirfoilShape(), iotype='out')

    def __init__(self):
        super(BezierAirfoilShape, self).__init__()

        self.fd_form = 'fd'

    def execute(self):

        print 'CPs', self.CPu
        self.spline_eval()

    def list_deriv_vars(self):

        inputs = ['CPu', 'CPl', 'CPle', 'CPte']
        outputs = ['afOut']

        return inputs, outputs

    def fit(self):


        iLE = np.argmin(self.afIn.points[:, 0])

        # lower side
        fit = FitBezier()
        fit.curve_in = Curve(points=self.afIn.points[:iLE + 1, :][::-1])
        fit.CPs = np.array([self.spline_CPs, np.zeros(self.spline_CPs.shape[0])]).T
        fit.constraints = np.ones((self.spline_CPs.shape[0], 2))
        fit.constraints[1, 0] = 0
        fit.fix_x = self.fix_x
        fit.execute()
        self.fit0 = fit.copy()

        self.spline_ps = fit.curve_out.copy()

        # upper side
        fit = FitBezier()
        fit.curve_in = Curve(points=self.afIn.points[iLE:, :])
        fit.CPs = np.array([self.spline_CPs, np.zeros(self.spline_CPs.shape[0])]).T
        fit.constraints = np.ones((self.spline_CPs.shape[0], 2))
        fit.constraints[1, 0] = 0
        fit.fix_x = self.fix_x
        fit.execute()
        self.fit1 = fit.copy()
        self.spline_ss = fit.curve_out.copy()

        self.CPl = self.spline_ps.CPs.copy()
        self.CPu = self.spline_ss.CPs.copy()
        self.nCPs = self.CPl.shape[0] + self.CPu.shape[0]
        self.CPle = 0.5 * (self.CPu[1, 1] - self.CPl[1, 1])
        self.CPte = 0.5 * (self.CPu[-1, 1] - self.CPl[-1, 1])
        if self.fd_form == 'complex_step':
            self.CPl = np.array(self.CPl, dtype=np.complex128)
            self.CPu = np.array(self.CPu, dtype=np.complex128)
            self.CPle = np.complex128(self.CPle)
            self.CPte = np.complex128(self.CPte)
        self.spline_eval()

    def spline_eval(self):
        """
        compute the Bezier spline shape given control points in CPs.
        CPs has the shape ((2 * spline_CPs.shape[0], 2)).
        """

        self.spline_ps.CPs = self.CPl
        if self.symmLE:
            self.spline_ps.CPs[1, 1] = -self.CPle
        if self.symmTE:
            self.spline_ps.CPs[-1, 1] = -self.CPte
        self.spline_ps.CPs[0] = np.array([0, 0])
        self.spline_ps.update()
        self.spline_ss.CPs = self.CPu
        if self.symmLE:
            self.spline_ss.CPs[1, 1] = self.CPle
        if self.symmTE:
            self.spline_ss.CPs[-1, 1] = self.CPte
        self.spline_ss.CPs[0] = np.array([0, 0])
        self.spline_ss.update()
        points = self.spline_ps.points[::-1]
        points = np.append(points, self.spline_ss.points[1:])
        points = points.reshape(points.shape[0]/2,2)
        self.afOut.initialize(points)


class FitBezier(Component):
    """
    Fit a Bezier curve to a 2D/3D discrete curve


    Parameters
    ----------
    curve_in : Curve-object
        Array of 2D or 3D points to be fitted ((n,j)) where n is the number of points
        and j is 2 or 3.
    CPs : array_like
        List containing initial guess of the control points. Spefify only interior control
        points since the end points of the Bezier control points will coincide with the
        end points of the curve.
    cons : array_like
        List containing simplified constraints for the control points ((n,j)) where j is 2 or 3.
    lsq_factor : float
        A parameter determining the initial step bound for the leastsq minimization
    lsq_epsfcn : float
        step length for the forward-difference approximation of the Jacobian
    lsq_xtol : float
        Relative error desired in the approximate solution.
    """

    curve_in = VarTree(Curve(), iotype='in')
    curve_out = VarTree(BezierCurve(), iotype='out')

    fix_x = Bool(False, iotype='in')
    nCPs = Int(iotype='in')
    CPs = Array(iotype='in', desc='Starting guess for control points')
    constraints = Array(iotype='in', desc='fixed/free (0 or 1) constraints for each control point'
                                         'e.g. [[0, 0], [1,0], [1,1], [0, 0]]')

    lsq_factor = Float(.01, iotype='in', desc='A parameter determining the initial step bound'
                                       'see scipy.optimize.leastsq for more info')
    lsq_epsfcn = Float(1e-4, iotype='in', desc='A suitable step length for the forward-difference approximation'
                                       'see scipy.optimize.leastsq for more info')
    lsq_xtol = Float(1.e-8, iotype='in', desc='Relative error desired in the approximate solution'
                                       'see scipy.optimize.leastsq for more info')

    def execute(self):

        # self.curve_in.redistribute(s=np.linspace(0, 1, self.curve_in.ni))

        if len(self.CPs) == 0:
            self.CPs = np.zeros((self.nCPs,self.curve_in.nd))
            for i in range(self.curve_in.nd):
                s = np.linspace(0., 1., self.nCPs)
                self.CPs[:, i] = np.interp(s, np.asarray(self.curve_in.s, dtype=np.float64),
                                              np.asarray(self.curve_in.points[:,i], dtype=np.float64))
        else:
            self.nCPs = self.CPs.shape[0]
            if np.sum(self.CPs[:, 1]) == 0.:
                self.CPs[:, 1] = np.interp(self.CPs[:, 0], np.asarray(self.curve_in.points[:, 0], dtype=np.float64),
                                                          np.asarray(self.curve_in.points[:, 1], dtype=np.float64))

        # anchor first and last CP to start/end points of curve
        self.CPs[0] = self.curve_in.points[0]
        self.CPs[-1] = self.curve_in.points[-1]

        # flatten the list
        self.parameters = list(self.CPs.flatten())

        # constraints
        if self.constraints.shape[0] == 0:
            self.constraints = np.ones((self.CPs.shape[0], self.curve_in.nd))
        # fix end points
        self.constraints[0] = 0.
        self.constraints[-1] = 0.

        # optionally fix all x-coordinates
        if self.fix_x:
            self.constraints[:, 0] = 0.

        # flatten constraints
        self.cons = self.constraints.flatten()

        # remove fixed parameters from list of parameters to be fitted
        # and add to fixedparams list
        self.fixedparams=[]
        self.delparams=[]

        for i in range(len(self.cons)):
            if self.cons[i] == 0:
                self.fixedparams.append(self.parameters[i])
                self.delparams.append(i)

        for i in range(len(self.delparams)):
            del self.parameters[self.delparams[i]-i]

        self.curve_out.dist_ni = self.curve_in.ni
        self.curve_out.fdist = self.curve_in.s.copy()
        self.iters = []
        res = leastsq(self.minfunc, self.parameters,full_output=1,factor=self.lsq_factor,
                      epsfcn=self.lsq_epsfcn,xtol=self.lsq_xtol)
        (popt, pcov, infodict, errmsg, ier) = res
        self.res = res
        self.popt = res[0]
        self._logger.info('Bezier fit iterations: %i' % infodict['nfev'])

    def minfunc(self, params):

        # gymnastics to re-insert fixed parameters into list passed to BezierCurve
        ii=0
        params=list(params)
        for i in range(len(self.cons)):
            if self.cons[i] == 0.:
                params.insert(i,self.fixedparams[ii])
                ii+=1
        params = np.array(params).reshape(len(params)/self.curve_in.nd,self.curve_in.nd)
        self.curve_out.CPs = params
        self.curve_out.update()
        self.curve_out.redistribute(s=self.curve_in.s)
        self.iters.append(self.curve_out.points.copy())

        res = np.zeros((self.curve_in.ni,self.curve_in.nd))
        for i in range(self.curve_in.ni):
            res[i,:] = ([(self.curve_in.points[i,j] - self.curve_out.points[i,j]) for j in range(self.curve_in.nd)])
        self.res = res.flatten()

        return self.res
