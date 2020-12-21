import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
from math import factorial
import scipy.fftpack


def s_n(number):
    a, b = '{:.4E}'.format(number).split('E')
    return '{:.5f}E{:+03d}'.format(float(a)/10, int(b)+1)


np.seterr(all="warn")
# Initials
lambda_zero = 0.775 * 10**(-4)
k_zero = 2 * np.pi / lambda_zero
c = 3 * 10**10
n_zero = 1
n_two = 3 * 10**(-19)
omega_zero = (c * k_zero) / n_zero
R_zero = 1
T_zero = 0.66 * 10**(-12)
P_NL = (lambda_zero**2) / (2 * np.pi * n_zero * n_two)
P_zero = P_NL
alpha_zero = 0.5
beta_zero = -20
beta_two = 2.2 * 10**(-31)
U_ion = 14.35
Kappa = 8
I_mp = 5.6e13
q = 1.6 * 10**(-19)
m = 9.31 * 10**(-31)
n_atm = 2.5e19
sigma_k = 2.88 * 10**(-99 / 8)
epsilon_zero = 8.84e-12
pi = 3.1415926535

ln = 0
alpha = 0
alph = alpha/4.343
gamma = (((omega_zero**2) * (n_zero**2) * n_two) / (4 * np.pi * c))
b2 = -20e-27
to = T_zero
ro = R_zero
Ld = (to**2)/np.absolute(beta_two)/33 #4.64 #3.117 #4.822


# T_z = np.arange(T_zero, 0.4e-12, -0.0013e-12)


tau = np.arange(-0.8e-12, 0.8e-12, 0.008e-12)
# tau = np.linspace(0.66e-12, 0.4e-12, 200)
r = np.linspace(-5.001, 5.001, 200)
Po = (16 * P_zero) / (c * n_zero * R_zero**2)
Ao = np.sqrt(Po)
dt = 0.003e-12
dr = 0.025
rel_error = 1e-5
z_max = 1.11
z_min = 0.1
h = 1500

op_pulse = [[0 for y in tau] for x in np.arange(z_min, z_max, z_min)]
pbratio = [0 for x in np.arange(z_min, z_max, z_min)]
phadisp = [0 for x in np.arange(z_min, z_max, z_min)]
number_density = [0 for x in np.arange(z_min, z_max, z_min)]

uaux = Ao * np.exp(-(1+1j*(-beta_zero))*(tau/to)**2 - (1+1j*(-alpha_zero))*(r/ro)**2)
uamp = Ao

for ii in np.arange(z_min, z_max, z_min):
    z = ii
    print(Ld)
    u = uaux[:]
    n = number_density[:]
    l = np.max(u.shape)
    fwhml = np.nonzero(np.absolute(u) > np.absolute(np.max(np.real(u))/2.0))
    fwhml = len(fwhml[0])

    dw_t = 1.0 / float(l) / dt * 2.0 * pi
    dw_r = 1.0 / float(l) / dr * 2.0 * pi

    w_t = dw_t*np.arange(-1 * l / 2.0, l / 2.0, 1)
    w_t = np.asarray(w_t)
    w_t = np.fft.fftshift(w_t)
    w_r = dw_r * np.arange(-1 * l / 2.0, l / 2.0, 1)
    w_r = np.asarray(w_r)
    w_r = np.fft.fftshift(w_r)

    u = np.asarray(u)
    u = np.fft.fftshift(u)
    spectrum = np.fft.fft(np.fft.fftshift(u))
    spectrum = np.array(spectrum, dtype=np.complex128)

    for jj in np.arange(h, z*Ld, h):
        spectrum = spectrum * np.exp((1 / (2.0 * 1j * k_zero)) *
                                     (-(h / 2.0) * np.power(w_r, 2) + k_zero * beta_two * np.power(w_t, 2) * (h / 2.0)))

        f = np.fft.ifft(spectrum)
        current = np.array(np.absolute((np.absolute(f) * (Po * c * n_zero / 8 * pi)) / uaux[99]))
        density_deriv = np.array(2 * pi * omega_zero / factorial(Kappa) * (current / I_mp)**Kappa * n_atm)
        omega_plasma_square = np.array((((4 * pi * q**2 * np.trapz(density_deriv)) / m) / c ** 2))
        f = f * (1 + ((1 / (2.0 * 1j * k_zero) *
                        (- gamma * np.power(np.absolute(f), 2) * h + h * omega_plasma_square / c ** 2 -
                       (h * 1j * (8.0 * pi * k_zero) / c) * (U_ion / np.power(np.absolute(f), 2))
                         * (2 * pi * n_atm * density_deriv * omega_zero / factorial(Kappa)) *
                         (((2 * P_zero)/(pi * R_zero**2))/I_mp)**Kappa))))
        f = np.array(f, dtype=np.complex128)

        spectrum = np.fft.fft(f)
        spectrum = spectrum * np.exp((1 / (2 * 1j * k_zero)) *
                                     (-(h / 2) * np.power(w_r, 2) + k_zero * beta_two * np.power(w_t, 2) * (h / 2)))

    f = np.fft.ifft(spectrum)
    I_rzt = np.absolute((np.absolute(f) * (Po * c * n_zero / 8 * pi)) / uaux[99])
    number_density_function = np.trapz(2 * pi * omega_zero / factorial(Kappa) * (I_rzt / I_mp)**Kappa * n_atm, dx=z*Ld)
    number_density[ln] = number_density_function
    op_pulse[ln] = np.absolute(f)
    print(s_n(I_rzt[99]))
    fwhm = np.nonzero(np.absolute(f) > np.absolute(np.max(np.real(f))/2.0))
    fwhm = len(fwhm[0])
    ratio = float(fwhm)/fwhml
    pbratio[ln] = ratio
    im = np.absolute(np.imag(f))
    re = np.absolute(np.real(f))
    div = np.dot(im, np.linalg.pinv([re]))
    dd = np.degrees(np.arctan(div))
    phadisp[ln] = dd[0]
    print("Progress: " + "> %d / " % ln + str(len(op_pulse)))
    ln = ln + 1

# Plots
print("\n\n> Plotting...")
trace_pulse_evolution = go.Surface(z=op_pulse, colorscale='Jet', y=np.arange(h, h * (ln + 1), h),
                                   x=np.linspace(r[0], r[-1], 200))
# trace_pulse_evolution = go.Scatter3d(z=np.arange(h, h * (ln + 1), h), y=tau,
#                                    x=r, marker=dict(size=12, color=op_pulse, colorscale='Jet', opacity=0.8))
trace_pulse_broad = go.Scatter(y=pbratio[0:ln], x=np.arange(1, ln+1, 1))
trace_phase_change = go.Scatter(y=phadisp[0:ln], x=np.arange(1, ln+1, 1))
# trace_number_density = go.Surface(z=number_density, colorscale='Jet', y=np.arange(h, h * (ln + 1), h))
trace_number_density = go.Scatter(y=number_density[0:ln], x=np.arange(h, h * (ln + 1), h))
trace_input_pulse = go.Scatter(y=np.absolute(uaux))
layout_input_pulse = go.Layout(
    autosize=False,
    width=500,
    height=400,
    title='Input Pulse',
    xaxis=dict(
        title='Time',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Amplitude',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
layout_number_2d = go.Layout(
    autosize=False,
    width=500,
    height=400,
    title='Number Density',
    xaxis=dict(
        title='z(cm)',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Number Density',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

layout_pulse_evolution = go.Layout(
    autosize=False,
    width=800,
    height=800,
    title='Pulse Evolution',
    scene=go.Scene(
        xaxis=go.XAxis(title='r(cm)'),
        yaxis=go.YAxis(title='z(cm)'),
        zaxis=go.ZAxis(title='Field Amplitude'))
)

layout_number_density = go.Layout(
    autosize=False,
    width=800,
    height=800,
    title='Number Density',
    scene=go.Scene(
        xaxis=go.XAxis(title='Time&Radius'),
        yaxis=go.YAxis(title='z(cm)'),
        zaxis=go.ZAxis(title='Number Density'))

)
layout_pulse_contour = go.Layout(
    autosize=False,
    width=600,
    height=400,
    title='Pulse Evolution Near Ionization',
    xaxis=dict(
        title='r(cm)',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='z(cm)',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
print(np.shape(op_pulse))
test_pulse = op_pulse[9:]
print(np.shape(test_pulse))
pulse_evolution_contour = go.Figure(data=go.Contour(z=test_pulse, x=r, y=np.arange(10*h, 12*h, h),
                                                    colorbar=dict(nticks=10, title="Amplitude", ticks='outside',
                                                                  ticklen=5, tickwidth=1,
                                                                  showticklabels=True,
                                                                  tickangle=0, tickfont_size=12)),
                                    layout=layout_pulse_contour)
pulse_evolution = go.Figure(data=[trace_pulse_evolution], layout=layout_pulse_evolution)

input_pulse = go.Figure(data=[trace_input_pulse], layout=layout_input_pulse)

number_evolution = go.Figure(data=[trace_number_density], layout=layout_number_2d)

plot([trace_phase_change], filename='./phase_change.html')
plot([trace_pulse_broad], filename='./pulse_broadening.html')
plot(pulse_evolution, filename='./pulse_evolution.html')
plot(input_pulse, filename='./input_pulse.html')
plot(number_evolution, filename='./number_density.html',)
plot(pulse_evolution_contour, filename='./pulse_evolution_contour.html')