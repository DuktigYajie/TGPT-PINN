import random

def random_point_in_diagonal_band(nu, N_x, width):
    r = random.uniform(0, 10)
    if r < 4:
        if r < 2:
            width = width/4
        t = random.randint(0, N_x)
        x0 = round(nu*t+3/4*(N_x-1))
        x = random.randint(x0-width/2,x0+width/2)% (N_x)
    elif r < 8:
        if r < 6:
            width = width/4
        t = random.randint(0, N_x)
        x0 = round(nu*t+1/4*(N_x-1))
        x = random.randint(x0-width/2,x0+width/2)% (N_x)
    else:
        x = random.randint(0, N_x)
        t = random.randint(0, N_x)
    return (t-1)*N_x+x-1

def random_point_in_initial(IC_pts, width):
    r = random.uniform(0, 10)
    if r < 4:
        x = random.randint((IC_pts-1)/4-width/2,(IC_pts-1)/4+width/2)
    elif r < 8:
        x = random.randint(3/4*(IC_pts-1)-width/2,3/4*(IC_pts-1)+width/2)
    else:
        x = random.randint(0, IC_pts)
    return x-1


