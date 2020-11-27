import numpy as np

x0min, x0max = -2.0, 2.0
x0num = 5
y0min, y0max = -2.0, 2.0
y0num = 5
kmax = 20000
base = 10
exp = -10

target = "main"

x0_ = np.linspace(x0min, x0max, x0num)
y0_ = np.linspace(y0min, y0max, y0num)

x0, y0 = np.meshgrid(x0_, y0_)


!make $target
for x0, y0 in zip(x0.flat, y0.flat):
    print(f"x0 = [{x0}, {y0}]")
    !./$target -k $kmax -b $base -e $exp $x0 $y0
