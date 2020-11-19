all:
	sage reproduce.py
	sage testing-z-shape.py
	sage tours_actually_run.py
	sage explain_why_more_security.py
	sage stddev_in_practice.py
	sage estimates.py

dsdgr20:
	rm -rf leaky-LWE-Estimator
	git clone https://github.com/lducas/leaky-LWE-Estimator.git
	cd leaky-LWE-Estimator; git checkout 948dbf0f89d57a712b442fec7fa0ecbe6736a57b
	patch leaky-LWE-Estimator/framework/utils.sage leaky-LWE-Estimator-utils.patch

aps15:
	rm -rf lwe-estimator
	git clone https://bitbucket.org/malb/lwe-estimator/
	cd lwe-estimator; git checkout 428d6ea75a1d0146f7b7bdfa6a26dbad2d5d12ca
	patch lwe-estimator/estimator.py lwe-estimator.patch

test:
	bash doctest.sh success_accumulator.py

clean:
	rm -rf plots
	mkdir -p plots/mixed
	mkdir -p plots/plain
	mkdir -p plots/progressive/vs-leaky
	mkdir -p plots/tour_maps
	mkdir -p plots/lwe-estimator-with-cn11
	mkdir -p plots/sample_variance

setup: clean dsdgr20 aps15
