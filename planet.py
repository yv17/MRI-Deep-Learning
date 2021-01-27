import numpy as np

def planet(I, alpha, TR, T1_guess, pcs=None,compute_df=False):
    '''Simultaneous T1, T2 mapping using phase‐cycled bSSFP.'''

    if compute_df:
        if pcs is None:
            pcs = np.linspace(0, 2*np.pi, I.size, endpoint=False)
        else:
            # Make sure we get phase-cycles as a numpy array
            pcs = np.array(pcs)
        assert pcs.size == I.size, ('Number of phase-cycles must match entries of I!')

    # Step 1. Direct linear least squares ellipse fitting to
    ## phase-cycled bSSFP data

    C = fit_ellipse(I.real, I.imag)
    C1, C2, C3, _C4, _C5, _C6 = C[:]
    assert C2**2 - 4*C1*C3 < 0, 'Not an ellipse!'

    #Step 2. Rotation of the ellipse to initial vertical conic form.
    xr, yr, Cr, _phi = _do_planet_rotation(I)
    I0 = xr + 1j*yr
    xc, yc = _get_center(Cr)

    # Look at it to make sure we've rotated correctly


    # Sanity check: make sure we got what we wanted:
    assert np.allclose(yc, 0), 'Ellipse rotation failed! yc = %g' % yc
    assert xc > 0, ('xc needs to be in the right half-plane! xc = %g'
                    '' % xc)


    # Step 3. Analytical solution for parameters Meff, T1, T2.
    # Get the semi axes, AA and BB
    A, B = _get_semiaxes(Cr)
    # Ellipse must be vertical -- so make the axes look like it
    if A > B:
        A, B = B, A
    A2 = A**2
    B2 = B**2

    # Decide sign of first term of b
    E1 = np.exp(-TR/T1_guess)
    aE1 = np.arccos(E1)
    if alpha > aE1:
        val = -1
    elif alpha < aE1:
        val = 1
    elif alpha == aE1:
        raise ValueError('Ellipse is a line! x = Meff')
    else:
        raise ValueError(
            'Houston, we should never have raised this error...')


    xc2 = xc**2
    xcA = xc*A
    b = (val*xcA + np.sqrt(xcA**2 - (xc2 + B2)*(A2 - B2)))/(xc2 + B2)
    b2 = b**2
    a = B/(xc*np.sqrt(1 - b2) + b*B)
    ab = a*b
    Meff = xc*(1 - b2)/(1 - ab)

    assert 0 < b < 1, '0 < b < 1 has been violated! b = %g' % b
    assert 0 < a < 1, '0 < a < 1 has been violated! a = %g' % a
    assert 0 < Meff < 1, '0 < Meff < 1 has been violated! Meff = %g' % Meff


    ca = np.cos(alpha)
    T1 = -1*TR/(
        np.log((a*(1 + ca - ab*ca) - b)/(a*(1 + ca - ab) - b*ca)))
    T2 = -1*TR/np.log(a)

    ## Step 4. Estimation of the local off-resonance df.
    if compute_df:

        costheta = np.zeros(pcs.size)
        for nn in range(pcs.size):
            x, y = I0[nn].real, I0[nn].imag
            t = np.arctan2(y, x - xc)

            if a > b:
                costheta[nn] = (np.cos(t) - b)/(b*np.cos(t) - 1)
            else:
                costheta[nn] = (np.cos(t) + b)/(b*np.cos(t) + 1)

        # Get least squares estimate for K1, K2
        X = np.array([np.cos(pcs), np.sin(pcs)]).T
        K = np.linalg.multi_dot((np.linalg.pinv(
            X.T.dot(X)), X.T, costheta))
        K1, K2 = K[:]

        theta0 = np.arctan2(K2, K1)
        df = -1*theta0/(2*np.pi*TR)
        return (Meff, T1, T2, df)

    return (Meff, T1, T2)

def _get_semiaxes(c):
    '''Get semiaxes of A and B'''

    A, B, C, D, E, F = c[:]
    B2 = B**2
    den = B2 - 4*A*C
    num = 2*(A*E**2 + C*D**2 - B*D*E + den*F)
    num *= (A + C + np.array([1, -1])*np.sqrt((A - C)**2 + B2))
    AB = -1*np.sqrt(num)/den

    return(AB[0], AB[1])

def _get_center(c):
    '''Compute center of ellipse.'''
    A, B, C, D, E, _F = c[:]
    den = B**2 - 4*A*C
    xc = (2*C*D - B*E)/den
    yc = (2*A*E - B*D)/den
    return(xc, yc)

def _rotate_points(x, y, phi, p=(0, 0)):
    '''Rotate points x, y through angle phi w.r.t. point p.'''
    x = x.flatten()
    y = y.flatten()
    xr = (x - p[0])*np.cos(phi) - (y - p[0])*np.sin(phi) + p[0]
    yr = (y - p[1])*np.cos(phi) + (x - p[1])*np.sin(phi) + p[1]
    return(xr, yr)

def _do_planet_rotation(I):
    '''Rotate complex pts to fit vertical ellipse centered at (xc, 0).'''


    # Represent complex number in 2d plane
    x = I.real.flatten()
    y = I.imag.flatten()

    # Fit ellipse and find initial guess at what rotation will make it
    # vertical with center at (xc, 0).  The arctan term rotates the
    # ellipse to be horizontal, then we need to decide whether to add
    # +/- 90 degrees to get it vertical.  We want xc to be positive,
    # so we must choose the rotation to get it vertical.
    c = fit_ellipse(x, y)
    phi = -.5*np.arctan2(c[1], (c[0] - c[2])) + np.pi/2
    xr, yr = _rotate_points(x, y, phi)

    # If xc is negative, then we chose the wrong rotation! Do -90 deg
    cr = fit_ellipse(xr, yr)
    if _get_center(cr)[0] < 0:
        # print('X IS NEGATIVE!')
        phi = -.5*np.arctan2(c[1], (c[0] - c[2])) - np.pi/2
        xr, yr = _rotate_points(x, y, phi)

    # Fit the rotated ellipse and bring yc to 0
    cr = fit_ellipse(xr, yr)
    yr -= _get_center(cr)[1]
    cr = fit_ellipse(xr, yr)

    ax = _get_semiaxes(c)
    if ax[0] > ax[1]:
        # print('FLIPPITY FLOPPITY!')
        xr, yr = _rotate_points(x, y, phi + np.pi/2)
        cr = fit_ellipse(xr, yr)
        if _get_center(cr)[0] < 0:
            # print('X IS STILL NEGATIVE!')
            phi -= np.pi/2
            xr, yr = _rotate_points(x, y, phi)
        else:
            phi += np.pi/2

        cr = fit_ellipse(xr, yr)
        yr -= _get_center(cr)[1]
        cr = fit_ellipse(xr, yr)
        # print(_get_center(cr))

    return(xr, yr, cr, phi)

def fit_ellipse(x, y):
    '''Ellipse fitting algorithm by Halir and Flusser.'''

    x = x.flatten()
    y = y.flatten()
    if x.size < 6 and y.size < 6:
        print('at least 6 points')

    D1 = np.stack((x**2, x*y, y**2)).T # quadratic pt of design matrix
    D2 = np.stack((x, y, np.ones(x.size))).T # lin part design matrix
    S1 = np.dot(D1.T, D1) # quadratic part of the scatter matrix
    S2 = np.dot(D1.T, D2) # combined part of the scatter matrix
    S3 = np.dot(D2.T, D2) # linear part of the scatter matrix
    T = -1*np.linalg.inv(S3).dot(S2.T) # for getting a2 from a1
    M = S1 + S2.dot(T) # reduced scatter matrix
    M = np.array([M[2, :]/2, -1*M[1, :], M[0, :]/2]) #premult by C1^-1
    _eval, evec = np.linalg.eig(M) # solve eigensystem
    cond = 4*evec[0, :]*evec[2, :] - evec[1, :]**2 # evaluate a’Ca
    a1 = evec[:, cond > 0] # eigenvector for min. pos. eigenvalue
    a = np.vstack([a1, T.dot(a1)]).squeeze() # ellipse coefficients
    return a

if __name__ == '__main__':
    pass