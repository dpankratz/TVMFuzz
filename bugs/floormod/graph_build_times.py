import matplotlib.pyplot as plt
import numpy as np
import ast
import sys

if __name__ == "__main__":
	if (len(sys.argv) < 2):
		print("Usage python graph_build_times.py build_times_file")
		sys.exit(1)
	f = open(sys.argv[1],"r")
	opt_avg = ast.literal_eval(f.readline())
	opt_std = ast.literal_eval(f.readline())
	nopt_avg = ast.literal_eval(f.readline())
	nopt_std = ast.literal_eval(f.readline())

	x = np.arange(max(len(nopt_avg),len(opt_avg)))
	plt.figure(figsize=(20,10))
	plt.bar(x-0.2,opt_avg,width = 0.4, yerr=opt_std,capsize = 5, label="Opt")
	plt.bar(x+0.2,nopt_avg, width = 0.4, yerr=nopt_std, capsize = 5, label="Non-opt")
	plt.yscale('log')
	plt.legend()
	plt.title("Build time")
	plt.savefig("figures/buildtime.png")
	plt.show()