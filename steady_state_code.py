amplitudes = []
amp_errors = []

for i in range(len(data[0])):
    w = (0.36 + i*0.1)*2*np.pi
    
    def driven_steady_state(t,A,phi,D):
        return A*np.cos(w*t+phi) + D

    steady_times = data[0][i][(len(data[0][i])//2):]
    steady_volts = data[1][i][(len(data[1][i])//2):]

    A_guess = np.abs(np.max(steady_volts)-np.min(steady_volts))
    low_bound = A_guess - A_guess/2
    high_bound = A_guess + A_guess/2

    params_initial = [A_guess,np.pi,-.15]
    param_bounds = ([low_bound,0,-0.3],[high_bound,2*np.pi,0])
    
    params, params_cov = curve_fit(driven_steady_state, steady_times, steady_volts,p0 = params_initial, bounds = param_bounds) # fit parameters
    params_error = np.sqrt([params_cov[0][0],params_cov[1][1],params_cov[2][2]])
    print(f"Best Fit Parameters and their Error for Trial {i+1}:")
    print("A = ", round(params[0],3), "+-", round(params_error[0],3))
    print("phi = ", round(params[1],3), "+-", round(params_error[1],3))
    print("D = ", round(params[2],3), "+-", round(params_error[2],3))
    print("Frequency (Fixed):",0.36 + i*0.1,"Hz")
    print("Chi-Squared Value for the Fit: ",np.sum(((driven_steady_state(data[0][i],params[0],params[1],params[2])-data[1][i])/0.0001)**2))

    amplitudes.append(params[0])
    amp_errors.append(params[1])

    plt.figure(figsize = (10,6))

    line1, = plt.plot(steady_times, steady_volts, color='black', linestyle='', marker = 'o', label='Data')

    x = np.linspace(steady_times[0], steady_times[-1], 500)
    line2, = plt.plot(x,driven_steady_state(x,params[0],params[1],params[2]), color='red', linestyle='-', marker='', label = "Line of Best Fit")

    # Name the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title(f"Voltage vs Time")
    plt.legend(loc = "upper right")

    plt.savefig(f"258_Lab4_Investigating_Resonance_Trial {i}")
    plt.show()
