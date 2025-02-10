list_of_times = []
list_of_volts = []

for i in range (2,14):
    times = pd.read_csv("/Users/ashoch/Downloads/MCGILL/PHYS 258/Lab4/resonance_investigation.csv", usecols = [f"Time (s) Run #{i}"])
    volts = pd.read_csv("/Users/ashoch/Downloads/MCGILL/PHYS 258/Lab4/resonance_investigation.csv", usecols = [f"Voltage, Ch B (V) Run #{i}"])
    times_array = np.array(times[f"Time (s) Run #{i}"].dropna())
    volts_array = np.array(volts[f"Voltage, Ch B (V) Run #{i}"].dropna())
    list_of_times.append(times_array)
    list_of_volts.append(volts_array)

data_chb  = [list_of_times,list_of_volts]

phases = []
phase_errors = []

for i in range(len(data_chb[0])):
    
    w = (0.36 + i*0.1)*2*np.pi
    
    def driven_steady_state(t,A,phi,D):
        return A*np.cos(w*t+phi) + D
        
    steady_times = data_chb[0][i][(len(data_chb[0][i])//2):]
    steady_volts = data_chb[1][i][(len(data_chb[1][i])//2):]

    params_initial = [0.015,np.pi,0]
    param_bounds = ([0.01,0,-0.01],[0.02,2*np.pi,0.01])
    
    params, params_cov = curve_fit(driven_steady_state, steady_times, steady_volts,p0 = params_initial, bounds = param_bounds,maxfev= 10000) # fit parameters
    params_error = np.sqrt([params_cov[0][0],params_cov[1][1],params_cov[2][2]])
    print(f"Best Fit Parameters and their Error for Trial {i+1}: (Drive voltage)")
    print("A = ", round(params[0],3), "+-", round(params_error[0],3))
    print("phi = ", round(params[1],3), "+-", round(params_error[1],3))
    print("D = ", round(params[2],3), "+-", round(params_error[2],3))
    print("Frequency (Fixed):",0.36 + i*0.1,"Hz")
    print("Reduced Chi-Squared Value for the Fit: ",np.sum(((driven_steady_state(steady_times,params[0],params[1],params[2])-steady_volts)/0.0001)**2)/len(data_chb[0][i]))

    phases.append(params[1])
    phase_errors.append(params_error[1])

    plt.figure(figsize = (10,6))

    line1, = plt.plot(steady_times, steady_volts,color='black', linestyle='', marker = 'o', label='Data')

    x = np.linspace(steady_times[0], steady_times[-1],500)
    line2, = plt.plot(x,driven_steady_state(x,params[0],params[1],params[2]), color='red', linestyle='-', marker='', label = "Line of Best Fit")

    # Name the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Driving Voltage (V)')
    plt.title("Driving Voltage vs Time")
    plt.legend(loc = "upper right")
    plt.show()

phases_drive = np.array(phases)
phases_response = np.array(phis)
