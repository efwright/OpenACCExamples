CC=pgcc
CXX=pgc++
FLAGS=-ta=tesla -Minfo=accel

matvecmul:
	$(CXX) -o matvecmul $(FLAGS) matvecmul.cpp

clean:
	rm -f matvecmul

