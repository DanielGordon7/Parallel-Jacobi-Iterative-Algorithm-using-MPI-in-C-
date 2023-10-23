run: compile
	mpiexec -np 4 ./pjacobi pa3_input_mat.txt pa3_input_vec.txt pa3_output.txt

run_many: compile
	mpiexec -np 4 ./pjacobi sample_input.txt sample_vec.txt my_output.txt
	mpiexec -np 9 ./pjacobi sample_input.txt sample_vec.txt my_output.txt
	mpiexec -np 16 ./pjacobi sample_input.txt sample_vec.txt my_output.txt
	mpiexec -np 25 ./pjacobi sample_input.txt sample_vec.txt my_output.txt

compile:
	mpicxx pjacobi.cpp -o pjacobi

clean:
	rm pjacobi