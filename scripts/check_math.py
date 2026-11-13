"""Small CPU-only independent checks of the manuscript's discrete algebra.

No training, checkpoint inference or GPU framework is imported. These checks
support the written derivations; they do not validate the continuum model.
"""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
from scipy.optimize import lsq_linear

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jaxwave.physics import propagate_numpy
source = ROOT / 'jaxwave/physics.py'
rng = np.random.default_rng(20261106)
rows = []
for steps in [6, 40, 160]:
    n, T, reg = 8, .2, 2e-4
    d = n*n
    c = rng.uniform(.65, 1.5, d)
    one = np.diag(-2*np.ones(n)) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
    D = (np.kron(one, np.eye(n)) + np.kron(np.eye(n), one)) * (n+1)**2
    B = c[:, None]**2 * D
    dt = T/steps
    # Independent dense polynomial recurrence for the entire operator.
    prev = np.eye(d)
    cur = prev + .5*dt*dt*B
    for k in range(1, steps):
        prev, cur = cur, 2*cur-prev+dt*dt*(B@cur)
    A = cur
    u = rng.uniform(-1, 1, d)
    r = rng.normal(size=d)
    z = np.stack([u.reshape(n,n), np.zeros((n,n))])
    reference = propagate_numpy(z[0], c.reshape(n,n), T=T, steps=steps).ravel()
    implementation_error = float(np.linalg.norm(A@u-reference)/np.linalg.norm(reference))
    adjoint_error = float(np.linalg.norm(A.T@r-(A@(c*c*r))/(c*c))/np.linalg.norm(A.T@r))
    symmetric = A*c[None, :]/c[:, None]
    symmetry_error = float(np.linalg.norm(symmetric-symmetric.T))
    weighted_norm = float(np.linalg.norm(symmetric, 2))
    norm = float(np.linalg.norm(A, 2))
    bound = float(c.max()/c.min())
    y = rng.normal(size=d)
    J = lambda x, target: float(np.sum((A@x-target)**2)+reg*np.sum(x*x))
    g = A.T@(A@u-y)+reg*u
    direction = rng.normal(size=d)
    eps = 1e-5
    fd = (J(u+eps*direction,y)-J(u-eps*direction,y))/(4*eps)
    gradient_error = float(abs(fd-g@direction)/max(1,abs(g@direction)))
    augmented = np.vstack([A,np.sqrt(reg)*np.eye(d)])
    opt = lsq_linear(augmented,np.r_[y,np.zeros(d)],bounds=(-1,1),tol=1e-12)
    opposite = lsq_linear(augmented,np.r_[-y,np.zeros(d)],bounds=(-1,1),tol=1e-12)
    assert opt.success and opposite.success
    odd_error = float(np.max(abs(opt.x+opposite.x)))
    actual_gap = (J(u,y)-J(opt.x,y))/(2*(y@y))
    certificate = float((g@u+np.linalg.norm(g,1))/(y@y))
    a,b = rng.uniform(-1,1,(2,d))
    f = .5*(a-b)
    mean_j = .5*(J(a,y)+J(b,-y))
    correction = .25*(np.sum((A@(a+b))**2)+reg*np.sum((a+b)**2))
    jensen_identity_error = float(abs(J(f,y)-(mean_j-correction)))
    assert implementation_error < 1e-10
    assert adjoint_error < 1e-10 and symmetry_error < 1e-10
    assert weighted_norm <= 1+1e-10 and norm <= bound+1e-10
    assert gradient_error < 1e-8 and odd_error < 1e-8
    assert -1e-10 <= actual_gap <= certificate+1e-10
    assert jensen_identity_error < 1e-10 and J(f,y) <= mean_j+1e-10
    rows.append(dict(n=n,steps=steps,CFL=float(dt*c.max()*(n+1)),
        dense_vs_source_relative_error=implementation_error,
        weighted_adjoint_relative_error=adjoint_error,
        similarity_symmetry_error=symmetry_error,weighted_operator_norm=weighted_norm,
        operator_norm=norm,operator_norm_bound=bound,
        half_objective_gradient_relative_error=gradient_error,
        optimum_sign_symmetry_max_error=odd_error,actual_normalized_half_gap=actual_gap,
        convex_gap_upper_bound=certificate,jensen_exact_identity_error=jensen_identity_error))
out = dict(status='passed',scope='Small independent dense NumPy/SciPy algebra checks; not continuum convergence or neural generalization evidence.',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,
    nominal_paper_CFL=.2/160*1.5*65,
    control_difference_bound_normalized_by_target_norm_at_gap_1e_5=float(np.sqrt(2e-5/2e-4)))
(ROOT/'outputs').mkdir(exist_ok=True)
(ROOT/'outputs/math_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
