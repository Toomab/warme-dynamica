import matplotlib.pyplot as plt
import numpy as np
from time import time
from scipy.integrate import quad
from scipy.stats import norm, poisson
from ipywidgets import interact
import ipywidgets as widget
from matplotlib.animation import FuncAnimation
#product and sum


def product_and_sum(a,b):
    return a*b, a+b
a=1.5
b=2.5
#two answers that returned are made p and s
p,s = product_and_sum(a,b)

print("Sum is:", s)
print("Product is:", p)


#raaklijn voor een functie:


#define f(x)=sin(x)
import numpy as np
def f(x):
    return np.sin(x)
print (f(0))
print(f(1))
#find slope of f(x) in the point a by taking the difference in y with two points x=a+epsilon and x=a-epsilon with a small epsilon
def determine_slope(a, epsilon):
    dy = f(a+epsilon)-f(a-epsilon)
    dx = (2*epsilon)
    return dy/dx
print (determine_slope(1,1e-7))
print (a)
#m=the slope of f(x) in point a, y=m*x+b so b=y-m*x, take this for x=a
def print_tangent(a, epsilon):
    m = determine_slope(a,epsilon)
    b = f(a)-m*a
    print("y={}x+{}".format(m,b))
#run function for a and epsilon of choice (the smaller the epsilon the more accurate)    
epsilon = 1e-7
a       = 0
print_tangent(a, epsilon)


#print as

print('The number given by the use is %.1f ' %a) #prints 1 dec float
print('The number given by the use is %d ' %a) #prints as integer
print('The number given by the use is %e ' %a) #prints with scientific notation


#take input a


a = input("Enter a number: ")
a = float(a) #change a to float
print("\nThe value of a is:", a)
print("a has the type:", type(a))


#arrays


array_evennr_1 = np.linspace(0,100,51)#0 to 100 but take 51 numbers
print(array_evennr_1)

array_evennr_2 = np.arange(0,101,2)#0 to 100 take every second number
print(array_evennr_2)


#lists and tuples


a = [2,3,5]#list, variable
b = (2,3,5)#tuple, (1) is int and (1,) is tuple, immutable so is object not variable
print(a[0]*2)#takes first number from list and multiplies by 2
print(b[0]*2)


#what can you do with arrays?


a = np.array([1,2,3,4,5])
a = np.append(a,6) #put 6 in as last variable
b = np.array([6,7,8,9,10])
b = np.delete(b,0) #delete first variable
c = np.concatenate((a,b)) #put together in a list, ordering variables like a,b
c = np.append(c,11)
print(c)
#can also np.append(a,b) to put two arrays together and np.sort(c) to sort entire array


#Graphs


#values
x = np.arange(0,10,1) #0 to 9, steps of 1
y = 2*x**2 #2x^2
z = 15*x+2 #15x+2

#graph
plt.rcParams['figure.dpi'] = 300 #used to set proper display resolution (pixels)

plt.figure(figsize=(5, 4)) #(w,h) in inches for the pdf
plt.xlabel("$x$ (m)") #label on x axis (x(m))
plt.ylabel("$t$ (s)") #label on y axis (t(s))

plt.plot(x,y,'k.',markersize='4',label='quadratic function') 
#plot y on x, line k, symbol ., size 4, label quadratic function
plt.plot(x,z,'r+',label='linear function') 
#plot z on x, line r (first letter for different colours), symbol +, label linear function

plt.legend(loc='upper left') #location of the legend

plt.xlim(0,10) #x values shown on axis
plt.ylim(0,180) #y values shown on axis
plt.grid() #adds a grid to the plot, not always appreciated by scientists.
plt.axhline(0, ls='-', lw=1, c='grey') #horizontal line on y=1
plt.axvline(0, ls=':', lw=1, c='grey') #vertical lines on x=1

plt.savefig('my_first_fig.pdf')
plt.show()


#creating a fitline


x = np.array([1.1, 3.4, 4.7, 5.3, 6.8])
y = np.array([1.05, 1.62, 1.95, 2.10, 2.47])
#fit to these arrays
def func(x,a,b): #3 variables in function have to be defined in order
    return a*x+b #how the three variables are ordered into a formula

x_test = np.linspace(0,1.2*max(x),1000) 
#x_test is 1000 variables in between 0 and 1.2*maximum value in the array x
y_test = func(x_test,.27,0.7) #three values defining y_test for every x_test
plt.xlim(0,1.1*max(x)) #how far the x axis reaches


#If


#== compares the two sides, true if both sides are the same
#also works with variables
#= causes error because not a conditional test
# can use mathematical expressions
#operators: >, <, >=, <=, != (not equal to)

#and is true if both are true
#or is true if one is true
#not is true if false

#operator precedence can cause different results: 
#(not false) or true is true, not (false or true) is false
#use brackets to show order

#else happens if if statements are false
#elif happens if if statements are false and it is true

a=-1
if a<6 and a>=0:
    if a>3:
        print("the value of a is 4 or 5")#prints if a>3
    else:
        print("the value of a is 0, 1, 2, or 3")#prints if a>3 is false
else:
    print("none of these are the case")#prints if a<6 or a>=0 are false


#while: runs until false

#a+=1 is a=a+1

def factorial(a):
    # f = our factorial result
    f = 1
    i = 2
    while(i<=a):
        f *= i #multiply f with i to make a new f
        i += 1 #add one to i and then go back to first step if i is smaller than or is equal to a
    return f
    
print(factorial(4))

s = 0
i = np.arange(1,11) #can also use range(11) instead of np.arange(1,11) to get numbers from 0 to 10 or range(1,11) to get 1 to 10
for i in i:
    s+=i
    print(s)

#range(A,B,C) gives numbers from A to B-1 with steps of C
#range can only count integers, not floats
#for loops can iterate over any iteratable object (also random number generators or arrays)

from numpy.random import randint

i = 0
found_five = False
max_tries = 10

while True:
    i += 1
    n = randint(0,30)
    if n == 5:
        found_five = True
        break
    if i >= max_tries: 
        break
        
if found_five:
    print("We found a 5 after", i, "tries")
else:
    print("We didn't find a 5 in the maximum number number of tries (", max_tries, ")")



s = 0
for i in randint(0,30,100): #random numbers between 0 and 30, do it 100 times
    if (i % 5) == 0: #% gives remainder after dividing i by 5
        continue #if i is divisible by 5 without remainder, start the next loop immediately
    s += i #s is the sum of all previous i

print("The sum of +/-100 random numbers between 0 and 30 excluding those that are divisible by 5 is:", s)


i=0
while i<1000000:
    if i*(i-10)==257024:
        print(i)
        break
    i+=1 #the sum has to go after, otherwise it starts checking the if statement at 1, meaning if i has to be 0 for the if statement to be true it does not give an answer


def leap_year(year):
    if (year % 4)==0:
        if (year % 100)==0:
            if (year % 400)==0:
                return True
            return False
        return True
    return False

# Check if your function works. The years 4, 2000, 2012 are examples of leap years,
# whereas the years 2021 and 2100 are not.
years = [1, 4, 100, 400, 2000, 2012, 2020, 2021, 2024, 2100]
for year in years:
    leap_year(year)



#autotestingscript
import numpy as np
test = []

for year in years:
    test.append(leap_year(year)) 

answercheck = [False, True, False, True, True, True, True, False, True, False]


#auto check
np.testing.assert_array_equal(test, answercheck, 'error', True)

#?????????


y = np.array([2, 2, 3, 4, 5, 4, 3, 8, 6, 4]) 

for i in range(1, len(y)-1): #this makes the range between the second and the second last numbers in array y 
    #(index means how many numbers to the left of them and len(y) is the index of the last number in y)
    if y[i-1] < y[i] and y[i+1] < y[i]: #if the number below and the number above is smaller
        print('index:', i, '\t value:', y[i])   #print how many numbers to the left and value. \t is a tab in between




password = 'practicum123' #password
tries = 0 #starts at 0 tries

from time import sleep

while True:
    input_pw = input('What is the password?') #input_pw is what you put in
    if input_pw == password:
        break #stops if password is correct
    if (tries+1)%3==0: #every third try, sleep for 60 seconds plus tries^3
        sleep(60+tries**3)
    tries += 1


#arrays


a = np.array([1,2,3,4,5]) #make an array
print(a)

a = np.array([1,2,3,4,5],dtype=float) #make an array of floats
print(a)

print(a[1]) #give 1st element of array

for i in range(len(a)):
    n = a[i]
    print('a[%d] is %d' % (i,n))

for n in a:
    print(n)

#in arrays you cant count further in indices than the length of the array
#you can count into negatives, which counts from the last value in the array towards the first
#a[-1]=a[last]

#a[0:5] means a new tuple with elements 0 to 5 from a
#can leave first or second  umber out. if no first number, defaults to start of array. if no end number, defaults to end of array
#a[0:5:1] the third number means how large the steps are. you can also take negative steps
#does not show error if you use the index for an element that doesnt exist
#arrays work like vectors but not when multiplying, then they just multiply every element from one with the same element of the other
#can get dotproduct with np.dot(a,b) and outer product with np.outer(a,b)
#cross can also be calculated with cross(a,b)
#< or other comparisons are true or false for each element

a = np.zeros(5)
print('zeros', a)
a = np.ones(5)
print('ones', a)
a = np.eye(5)#diagonal line of 1s
print(a)
a = np.tri(5) #diagonal with all 1s below
print(a)


#numpy


#np.linspace(start,end,number of points) note: number of points include 1st point so 1 extra than expected
#np.arange(start,end (not included),spacing) like range(start,end+1,spacing) but range only uses integers
#np.geomspace() produces geometrically spaced points so they are placed evenly on a log scale
#np.random.random(number of elements) generates random numbers between 0 and 1
#np.random.normal(avg,standard deviation,amount of elements)
#np.round() rounds to whole numbers


#plotting


#data

raw = np.random.normal(7.5,1,300)
rounded_grades = np.round(raw*2)/2
print(rounded_grades)

#plot

bins = np.arange(np.min(rounded_grades),np.max(rounded_grades)+0.5,0.5) #make an array from min to max including max with steps of 0.5

plt.figure()
plt.hist(rounded_grades, bins) 
#histogram showing how many elements of the rounded_grades array are equal to each possible grade
plt.xlabel('Grade') 
plt.ylabel('#') 
plt.show()


#numpy arrays can be multidimensional vectors
#np.zeros([rows,columns])
#len(vector) only gives number of rows
#vector.shape gives (rows,columns)
#"\n" is a line skip

#vector[row index,column index] where 0 is the index of the first element on rows and columns
#vector(row index,:) selects an entire row, where you can then use [element1,element2,...,elementlast] to change elements
#this can also be used to extract rows or columns
#blocks can be extracted with vector[starting row index:ending row index,starting column index:ending column index]

#np.eye(rows and columns) gives identity matrix
#np.eye(rows and columns,k=how much diagonal is shifted up)

#get sum of elements with dotproduct with a vector of ones
#cumulative array can be achieved with dotproduct of np.tri(columns and vectors) and the vector
#or with np.cumsum(array)

#or use
sum_a=0
cumsum=0
for x in a:
    sum_a += x
    cumsum = np.append(cumsum,sum_a)
#to get sum and cumulative

#np.average(array) gives average
#np.std(array,ddof=...)??? caluculates standard deviation
#np.sqrt(array)
#np.min(array) or max
#np.argmin(array) gives the index of the minimum value
#can always input (array operation) instead of array: array*2 for example

#vectorised means optimised

#print("There are %d fives" % np.sum(nums == 5))
#%d refers to the part after the other %, instead of haveing to name it

#np.sum can convert true or false into numbers

#.astype('int') converts into integer

#if you make a function, can also vectorise with np.frompyfunc(function,1,1)


def f(x):
    return np.exp(x**2) #e^x^2

x=np.linspace(0,1,1000) #array with 1000 elements between 0 and 1

plt.figure()
plt.plot(x,f(x))
plt.ylim(0) #only place lower limit because otherwise the lower y would not be shown
plt.xlim(0,1)
plt.show()
print(np.max(f(x))) #maximum of f(x). this means x is between 0 and 1 because of the linspace created before


#calculating an integral with python (montecarlo)


N = int(1e6)   #number of points that we are going to use in our calculation
x_a = 0
x_b = 1
y_a = 0
y_b = 3
#Generate samples:
random_x = np.random.uniform( x_a,x_b ,size=N) #N points between x_a and x_b with a uniform spread
random_y = np.random.uniform(y_a ,y_b ,size=N)

t1 = time()

s = 0

for i in range(len(random_x)): #if i is a number corresponding to an index of random_x
    if random_y[i] <= np.exp(random_x[i]**2): 
        #if the element of random_y with index i is smaller than or equal to e^x^2 for x=element of random_x with index i
        s+= 1 #then s=s+1 so s is the amount of elements of random_y below e^x^2

solution_integral = (x_b-x_a)*(y_b-y_a)*s/N #dx*dy*s/N
#=area of square formed by points x_b, x_a, y_b and y_a times the points below e^x^2 divided by total amount of points
#=area*what percentage of the area is taken up by e^x^2
print('The solution of the integral is %.6f' %(solution_integral))            
print('Time for calculation: %.3f s' %(time()-t1))

#check

Area = np.abs(x_b-x_a)*np.abs(y_b-y_a) #area
err_solution = Area*np.sqrt(s)/N 
#estimated error of our solution, which is the area of our random uniform box divided by the number of points used in the calculation times the poisson uncertainty

scipy_solution = quad(f,0,1) #calculating integral between x=0 and x=1 using quad function from scipy, returns: value of integral and estimated error

abs_difference = np.abs(solution_integral-scipy_solution[0]) #determine absolute difference between solutions
#scipy_solution consists of 2 elements: the solution and the error

assert abs_difference <= 2*np.sqrt(err_solution**2 + scipy_solution[1]**2), 'The results are not in agreement'  #check if the values are in agreement 

print('Our solution: %.3f +/- %.3f' %(solution_integral, err_solution))
print('Scipy solution: %.14f +/- %.14f' %(scipy_solution[0], scipy_solution[1]))
print('Difference between the solutions: %.6f' %(abs_difference))

t1 = time()

s = np.sum(random_y<=np.exp(random_x**2)) #number of points where y is below the function
#the part within brackets produces an array where every element of random_y is true or false, then the amount of true is added up

solution_integral = (x_b-x_a)*(y_b-y_a)*s/N

print('The solution of the integral is %.6f' %(solution_integral))            
print('Time for calculation: %.3f s' %(time()-t1))


#can extract values that satisfy the condition with new_array=array[array{condition}]


random_x = np.random.uniform(0,1,1000) #uniform random of 1000 points
random_y = np.random.uniform(0,3,1000) 

points_within_area = np.array([random_x[random_y <= np.exp(random_x**2)],random_y[random_y <= np.exp(random_x**2)]])
points_outside_area = np.array([random_x[random_y > np.exp(random_x**2)],random_y[random_y > np.exp(random_x**2)]])
#arrays where the first part is x values where the condition is satisfied and the second row is the y values
#np.array([x,y])

plt.figure(figsize=(6,4)) #not necessary
plt.plot(points_outside_area[0],points_outside_area[1], '.',c='red') #points[0] gives x value and points[1] gives y
plt.plot(points_within_area[0],points_within_area[1], '.', c='green')
#plt.plot(x,y,'dot shape', c='colour')
#can also use plt.plot(x,y,'colourdotshape',markersize='...',label='...')
plt.plot(x,f(x),linewidth=3,label='f(x)') #plot f(x)
plt.xlim(0,1)
plt.ylim(0,3)
plt.ylabel('f(x)')
plt.xlabel('x')
plt.legend()
plt.show()


def moving_avg(p, k):
    
    F = np.empty(len(p)-(k-1))
    for i in range(len(p)-(k-1)):
        F[i] = (1/k)*np.sum(p[i:i+k]) #sum(p[i to i+k-1])/k
        
    return F

Z = np.arange(4,20) #array from 4 to 20(not included)
print(Z)
print(moving_avg(Z, 3)) #computes average of every 3 consecutive elements

x_raw=np.arange(0,101,1)
x_even2 = x_raw[x_raw % 2 == 0] #x_raw%2==0 converts x_raw into a bolean array, where a value is true if it is divisible by two
#x_even2 is then equal to every value of x_raw that is divisble by two in order




#oefentoets



#vraag 1:
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
x_raw=np.array(range(0,101))
x_even=np.zeros(51,dtype=int)
for i in range(0,101):
  if i%2==0:
    x_even[int(i/2)]=x_raw[i]
def func(x,a):
  return a*x**2

y=func(x_even,2.0)
for i in range(51):
  if func(20,2.0)==y[i]:
    y[i]=2500

a=curve_fit(func,x_even,y)
print("the best value for a is:",a[0],"+-",a[1])
x_test=np.arange(0,101,0.1)


plt.figure()

plt.plot(x_even,y,'k.')
plt.plot(x_test,func(x_test,a[0]),'r-')

plt.show()

#uncertainty


# --- Calculus Approach ---
#Z_unc = absolute(dZ/dA) * uncertainty A

# --- Functional Approach ---
#relative uncertainty of A = uncertainty A/A
#relative uncertainty of Z = exponent * relative uncertainty of A
#uncertainty of Z = relative uncertainty of Z * Z


#uncertainty, standard deviation and mean

#mean= sum(all x)/N of points
#standard deviation = sqrt(sum((every point x-avg x(1 for every point)^2)/(N-1))
#uncertainty=standard deviation/sqrt(N)

#weighing factor=1/u^2
#weighted avg = (every x*their weighing factor)/sum(every weighing factor)
#new uncertainty = sqrt(1/sum(weighing factors))

#automatically round to any number of significant digits:
def round_sig(x, sig=1):
    if x == 0:
        return 0
    return np.sign(x) * round(abs(x), -int(np.floor(np.log10(abs(x)))) + (sig - 1))

#probability of a point= e^(-(x-avg)^2/2*std^2)/std*sqrt(2*pi)


# Maken van 1000 random punten

def update(average_value,std_value):

    x = np.random.normal(average_value,std_value,1000)

    # Bereken van gemiddelde waarde en standaard deviatie
    av_x = np.array([])
    std_x = np.array([])
    for i in range(1,len(x)-1):
        av_x = np.append(av_x,np.mean(x[:i]))
        std_x = np.append(std_x,np.std(x[:i],ddof=1))

    # Plotten van de waarden als functie van N
    plt.clf() #clear previous figures
    fig, axs = plt.subplots(1,3,figsize=(15, 5)) #creates 1 row and 3 columns of figures
    axs[0].plot(x,'k.',markersize=1) #figure with index 0
    axs[0].set_xlabel('N')
    axs[0].set_ylabel('')

    axs[1].plot(av_x,'k.',ms=1)
    axs[1].set_xlabel('N')
    axs[1].set_ylabel('average of x as function of N')

    axs[2].plot(std_x,'k.',ms=1)
    axs[2].set_xlabel('N')
    axs[2].set_ylabel('standard deviation of x as function of N')

    plt.show()

interact(update, average_value=widgets.FloatSlider(min=-3, max=3, step=.5, value=0), #creates slider with min -3 max 3 step 0.5 and initial value 0
         std_value=widgets.FloatSlider(min=0, max=10, step=1, value=1))


#poisson:

#chance(x=k)=((avg^k)*e^-avg)/factorial(k)
#std=sqrt(avg)

#an outlier may be removed if it has a lower probability that 1/2*N
#or if N*Pout<0.5
#Pout=2erf(xout,avgx,stdx)

#errorfunctions

#erf(x) = 2/sqrt(pi)*integral(e^-t^2dt)up=x, low=0
#also erf(x) in scipy.special.erf

#erf(x,avgx,stdx)=0.5(1+erf((x-avgx)/sqrt(2)*stdx))
#also norm.cdf in scipy.stats.norm.cdf


from scipy.stats import norm

#P is the dataset (a numpy array)
x_out = np.max(P) #In this case, could also have been other outliers
x_mean = np.mean(P)
x_std = np.std(P,ddof=1)

#Use the Erf function
Q = norm.cdf(x_out,x_mean,x_std)

#You could have also defined the Erf on your own

#Check if it is a high 'outlier'
if Q > 0.5:
    Q = (1-Q)

#Use Chauvenets criterion
C =  2 * len(P) * Q

if C < 0.5:
    print('The value can be discarded.')
else:
    print('The value cannot be discarded.')


    #????


#sims

# Definieer een class voor een deeltje. 
# Een class beschrijft alleen het type eigenschappen dat een ding heeft, bijvoorbeeld: in de beschrijving van de class zeg je:
# - een deeltje heeft een positie.
# - een deeltje heeft een functie die, wanneer aangeroepen, de positie van dat deeltje bijwerkt.
# Merk op dat de class zelf geen deeltje is! 
# Een object dat tot een bepaalde class behoort, wordt een instant van die class genoemd. 
# Je kunt meerdere instants van een class hebben.

class ParticleClass:
    def __init__(self, m, v, r, R,c):
        self.m = m                         # massa van het deeltje
        self.v = np.array(v, dtype=float)  # snelheids vector
        self.r = np.array(r, dtype=float)  # positie vector
        self.R = np.array(R, dtype=float)  # radius van het deeltje
        self.c = np.array(c, dtype=float)  #colour of the particle

    def update_position(self,dt):
        """Werk de positie van het deeltje bij op basis van zijn snelheid en tijdstap dt."""
        self.r += self.v * dt

print(ParticleClass)



#maak 10 particles met willekeurige snelheid tussen -5 en 5, en plot ze
particle_array2 = []

for i in range(10): #10 particles
    color=np.random.rand(3,) #kleur
    particle_array2.append(ParticleClass(m=1.0, v=np.random.rand(2)*10-5, r=[0.0, 0.0],R=1.0,c=color)) #definituie en in array

def plotpos(x): #plot positie voor update x
    for particle,particle_object in enumerate(particle_array2):
        particle_object.update_position(x)
        plt.plot(particle_object.r[0],particle_object.r[1],marker='.',color=particle_object.c)


plt.figure()
for x in range(10): #plot 10 updates
    plotpos(x)
plt.show()



#in box with walls and acceleration

# Maken van de class met versnelling
class ParticleClass:
    def __init__(self, m, v, r, R):
        self.m = m                  # mass of the particle
        self.v = np.array(v, dtype=float)  # velocity vector
        self.r = np.array(r, dtype=float)  # position vector
        self.R = np.array(R, dtype=float)  # radius of the particle

    def update_position(self):
        self.r += self.v * dt
    
    def update_velocity(self, a, i):
        self.v[i] += a[i]*dt


# Simulation parameters
dt = 0.1         # time step
a =[0,-9.81]   # acceleration
num_steps = 500  # number of time steps
particle = ParticleClass(m=1.0, v=[5.0, 3.0], r=[0.0, 0.0],R=1.0)  
lower=-10
upper=10
# Create the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(lower, upper)
ax.set_ylim(lower, upper)
ax.set_aspect('equal')
ax.set_title("Particle Animation")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Create the particle as a red dot
dot, = ax.plot([], [], 'ro', markersize=10)

# Initialization function for animation
def init():
    dot.set_data([], [])
    return dot,

def bounce(x):
    if particle.r[x]>upper:
        particle.v[x]=-particle.v[x]
    if particle.r[x]<lower:
        particle.v[x]=-particle.v[x]

# Update function for each frame
def update(frame):
    particle.update_position()
    particle.update_velocity(a,0)
    particle.update_velocity(a,1)
    dot.set_data([particle.r[0]], [particle.r[1]])
    bounce(0)
    bounce(1)
    return dot,

# Create animation
ani = FuncAnimation(fig, update, frames=range(200), init_func=init, blit=True, interval=50)

# For Jupyter notebook:
from IPython.display import HTML
HTML(ani.to_jshtml())