diff_errors = np.sqrt(np.array(phi_errors)**2 + np.array(phase_errors)**2)*100
phase_diffs = -np.unwrap(phases_response-phases_drive)
phase_diffs = phase_diffs + np.pi

def cosine_fit(w,gamma):
    return np.arccos((omega_res**2-w**2)/(np.sqrt((omega_res**2-w**2)**2+(w*gamma)**2)))

params_initial = [0.4]
param_bounds = ([0.25],[0.75])
params, params_cov = curve_fit(cosine_fit, frequencies, phase_diffs,p0 = params_initial, bounds = param_bounds) # fit parameters
params_error = np.sqrt([params_cov[0][0]])


def gamma_error_propagation(w, gamma, gamma_error):
    numerator = -gamma * w
    denominator = ((omega_res**2 - w**2)**2 + (w * gamma)**2) ** (3/2)
    dphi_dgamma = np.abs(numerator / denominator)
    return dphi_dgamma * gamma_error  # Propagated uncertainty

# Compute propagated uncertainty in phase difference
propagated_errors = gamma_error_propagation(frequencies, params, params_error)
total_errors = np.sqrt(diff_errors**2 + propagated_errors**2)

print("Best Fit Parameters and their Errors:")   
print("gamma = ", round(params[0],3), "+-", round(params_error[0],3))
print("Reduced Chi-Squared Value for the Fit: ",np.sum(((cosine_fit(frequencies,params[0])-phase_diffs)/total_errors)**2)/(len(frequencies)-1))

# Name the axes
plt.figure(figsize = (10,6))
line1 = plt.errorbar(frequencies, phase_diffs, yerr = total_errors, color='black', linestyle = '', marker = 'o', label='Data with Error Bars')

x = np.linspace(frequencies[0], frequencies[-1], 500)
line2, = plt.plot(x,cosine_fit(x,params[0]), color='red', linestyle='-', marker='', label = "Line of Best Fit")

plt.title("Phase Difference as a Function of Frequency")
plt.xlabel("Driving Frequency (rad/s)")
plt.ylabel("Phase Difference (rad)")
plt.legend(loc = "upper left")
plt.savefig("phase(w)")
plt.show()
