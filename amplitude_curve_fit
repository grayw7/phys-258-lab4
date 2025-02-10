frequencies = 2*np.pi*np.array([0.36,0.46,0.56,0.66,0.76,0.86,0.96,1.06,1.16,1.26,1.36,1.46])
amp_array = np.array(amplitudes)
err_array = np.array(amp_errors)

omega_res = 2*np.pi*0.8654

def A(omega,A0,gamma):
    return A0/np.sqrt((omega_res - omega)**2 + (gamma*omega)**2)

params_initial = [0.05,0.02]
param_bounds = ([0,0],[0.1,0.05])
params, params_cov = curve_fit(A, frequencies, amp_array,p0 = params_initial, bounds = param_bounds) # fit parameters
params_error = np.sqrt([params_cov[0][0],params_cov[1][1]])

def propagated_error_A(frequencies, A0, gamma, A0_error, gamma_error):
    # Partial derivative of A with respect to A0
    dA_dA0 = 1 / np.sqrt((omega_res - frequencies)**2 + (gamma * frequencies)**2)
    # Partial derivative of A with respect to gamma
    dA_dgamma = -A0 * frequencies / ((omega_res - frequencies)**2 + (gamma * frequencies)**2)**(3/2)
    
    # Propagating errors
    propagated_A0_error = dA_dA0 * A0_error
    propagated_gamma_error = dA_dgamma * gamma_error
    
    # Total propagated error
    total_error = np.sqrt(propagated_A0_error**2 + propagated_gamma_error**2)
    return total_error

# Calculate the propagated errors for each frequency
propagated_errors = propagated_error_A(frequencies, params[0], params[1], params_error[0], params_error[1])

# Calculate the total errors (including the original uncertainties in the data)
total_errors = np.sqrt(err_array**2 + propagated_errors**2)

print("Best Fit Parameters and their Errors:")   
print("A0 = ", round(params[0],3), "+-", round(params_error[0],3)) 
print("gamma = ", round(params[1],3), "+-", round(params_error[1],3))
print("Reduced Chi-Squared Value for the Fit: ",np.sum(((A(frequencies,params[0],params[1])-amp_array)/total_errors)**2)/(len(frequencies)-1))

# Name the axes
plt.figure(figsize = (10,6))
line1, = plt.plot(frequencies, amp_array, color='black', linestyle='', marker = 'o', label='Data')

x = np.linspace(frequencies[0], frequencies[-1], 500)
line2, = plt.plot(x,A(x,params[0],params[1]), color='red', linestyle='-', marker='', label = "Line of Best Fit")

plt.title("Amplitude as a Function of Frequency")
plt.xlabel("Angular Frequency (rad/s)")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.savefig("A(w)")
plt.show()
