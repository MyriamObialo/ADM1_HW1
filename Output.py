##### INTRODUCTION ######
#Say "Hello, World!" With Python 1
if __name__ == '__main__':
    print("Hello, World!")

#Arithmetic Operators 2
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

#Python: Division 3
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

#Loops 4
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i**2)

#Print Function 5
if __name__ == '__main__':
    n = int(input())
    for i in range(1,n+1):
        print(i, end = "")

#Python If-Else 6
import math
import os
import random
import re
import sys
if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 !=0: 
        print('Weird')
    elif n%2 == 0 and n in range (2,6):
        print('Not Weird')
    elif n%2 == 0 and n in range (6,21):
        print('Weird')
    elif n%2 == 0 and n > 20:
        print('Not Weird')

#Write a function 7 
def is_leap(year):
    leap = False
    if year % 400 ==0:
        leap = True
    elif year % 100 != 0 and year % 4 ==0:
        leap = True
    return leap

####### STRINGS #######

#What's Your Name? 1
def print_full_name(first, last):
    print("Hello " + first +' '+ last+ "! You just delved into python.")


#Mutations 2
def mutate_string(string, position, character):
    listas= list(string)
    listas[position] = character
    final= "".join(listas)
    return final
    
#sWAP cASE 3
def swap_case(s):
    string=""
    s_list=list(s)
    for i in s_list:
        if i==i.upper():
            string+=i.lower()
        elif i==i.lower():
            string+=i.upper()
    return(string)

#String Split and Join 4
def split_and_join(line):
    return (line.replace(" ", "-"))

#Find a string 5
def count_substring(string, sub_string):
    count=0
    for x in range(0,len(string)):
        if string.find(sub_string,x) == x:
            count+=1
            
    return count

#Text Wrap 6
def wrap(string, max_width):
    s=(textwrap.wrap(string,max_width))
    return "\n".join(s)
    
# String Validators 7 
if __name__ == '__main__':
    s = input()
    print(any(i.isalnum() for i in s))
    print(any(i.isalpha() for i in s))    
    print(any(i.isdigit() for i in s))
    print(any(i.islower() for i in s))
    print(any(i.isupper() for i in s))

#String Formatting 8
def print_formatted(number):
    for i in range(1, number+1):
        width= len(bin(number)) -2
        print(f"{i:>{width}} {oct(i)[2:]:>{width}} {hex(i)[2:].upper():>{width}} {bin(i)[2:]:>{width}}")

#Capitalize! 9
def solve(s):
    result=[]
    if 0<len(s)<1000:
        for i in s.split(" "):
            result.append(i.capitalize())
        return ' '.join(result) 

#Text Alignment 10

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#Designer Door Mat 11
l= input().split(' ')
rows = int(l[0])
cols = int(l[1])
c='-'
p='.|.'
u= cols - 3
t= u//2
w= 'WELCOME'
m= int((cols - len(w))/2)

a= 0 
b=1
for i in range(rows//2):
    print((c*(t+a)+ p*b + c*(t+a)))
    a-= 3

#Alphabet Rangoli 12
def print_rangoli(size):
    length=((size * 2) - 1)
    base = [abs(size - x - 1) for x in range(length)]
    for i in range(length):
        new = [x + base[i] for x in base]
        prepend=''
        for j in new:
            if j > size - 1:
                print(prepend+'-', end='')
            else:
                print(prepend+chr(ord('a')+j), end='')
            prepend='-'
        print('')
        

#Merge the Tools! 13
def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        sub = string[i:i+k]
        seen = []
        for char in sub:
            if char not in seen:
                seen.append(char)
            
        print(''.join(seen), end = "\n")

#The Minion Game 14
def minion_game(string):
    vowels= 'AEIOU'
    k_score= 0
    s_score= 0
    length = len(string)

    for i in range(length):
        if string[i] in vowels:
            k_score+=length - i
        else:
            s_score +=length - i

    if k_score >s_score:
        print(f"Kevin {k_score}")
    elif s_score >k_score:
        print(f"Stuart {s_score}")
    else:
        print("Draw")

###### BASIC DATA TYPES #######

#Find the Runner-Up Score! 1
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    no_duplicates= set(arr)
    l=list(no_duplicates)
    l.remove(max(l))
    print(max(l))

#Lists 2
if __name__ == '__main__':
    N = int(input())
    r= list()
    for x in range(N):
        l= list(input().split())
        for x in range(1,len(l)):
            l[x]=int(l[x])
        if l[0]=="append":
            r.append(l[1]) 
        elif l[0]=="insert":
            r.insert(l[1],l[2])
        elif l[0]=="print":
            print(r)
        elif l[0]=="remove":
            r.remove(l[1])
        elif l[0]=="sort":
            r.sort()
        elif l[0]=="pop":
            r.pop()
        elif l[0]=="reverse":
            r.reverse()

#List Comprehensions 3
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    s=list()
    for i in range(x+1):
        for u in range(y+1):
            for t in range(z+1):
                if (i+u+t) != n:
                    s.append([i,u,t])
print(s)

#Tuples 4
n = int(input())
integer_list = tuple(map(int, input().split()))
print(hash(integer_list))

# Finding the percentage 5
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    if query_name in student_marks: 
        b=list(student_marks[query_name])
        print(format(sum(b)/3,".2f"))

#Nested Lists 6
if __name__ == '__main__':
    d=dict()
    for _ in range(int(input())):
        name = input()
        score = float(input())
        d[name]=score
    scores=sorted(set(d.values())) 
    scores.remove(scores[0])
    value = {i for i in d if d[i]== scores[0]}
    for x in sorted(value):
        print(x)

###### SETS #######

# Introduction to Sets 1
def average(array):
    l_array= set(array)
    result= float(sum(l_array))/len(l_array)
    return result

#Symmetric Difference 2
n=int(input())
set1=list(map(int,input().split()))
y=int(input())
set2=list(map(int,input().split()))
l=list()
if len(set1)<= n and len(set2)<=y:
    for x in set(set1) ^ set(set2):
        l.append(x)
        l.sort()
    for x in l:
        print (x)

#Set.add() 3
N= int(input()) 
N_set=set()
for i in range(N):
    s=input()
    N_set.add(s)
list5=len(list(N_set))
print(list5)

#Set .discard(), .remove() & .pop() 4
n = int(input())
s = set(map(int, input().split()))

N=int(input())
for x in range(N):
    commands=input().split()
    if commands[0]=='pop':
        s.pop()
    elif commands[0]=='remove':
        s.remove(int(commands[1]))
    elif commands[0]=='discard':
        s.discard(int(commands[1]))

print(sum(s))

#Set .union() Operation 5
english=int(input())
students_e=set(map(int,input().split()))
french=int(input())
students_f=set(map(int,input().split()))

total= students_e | students_f
print(len(total))

#Set .intersection() Operation 6
english=int(input())
students_e=set(map(int,input().split()))
french=int(input())
students_f=set(map(int,input().split()))

total= students_e & students_f
print(len(total))

#Set .difference() Operation 7
english=int(input())
students_e=set(map(int,input().split()))
french=int(input())
students_f=set(map(int,input().split()))

total= students_e - students_f
print(len(total))

#Set .symmetric_difference() Operation 8
english=int(input())
students_e=set(map(int,input().split()))
french=int(input())
students_f=set(map(int,input().split()))

total= students_e ^ students_f
print(len(total))

#Set Mutations 9
A=int(input())
setA=set(map(int,input().split()))
N=int(input())
for i in range(0,N):
    s=input().split()
    other_set = set(map(int, input().split()))
    if s[0]=="intersection_update":
        setA.intersection_update(other_set)
    if s[0]=="update":
        setA.update(other_set)
    if s[0]=="symmetric_difference_update":
        setA.symmetric_difference_update(other_set)
    if s[0]=="difference_update":
        setA.difference_update(other_set)
print(sum(setA))

#The Captain's Room 10
k=int(input())
number_list=list(map(int,input().split()))
d=dict()
for x in number_list:
    if x in d:
        d[x]+=1
    else:
        d[x]=1
for x in d:
    if d[x]!= k:
        print(x)

#Check Subset 11
T = int(input())

for i in range(T):
    N = int(input())
    setA = set(map(int,input().split()))
    B = int(input())
    setB = set(map(int,input().split()))
    if (setA & setB) == setA :
        print("True")
    else:
        print("False")

#No Idea! 12
happiness=0
n, m =map(int,input().split())

n_array=list(map(int, input().split()))
a=set(map(int, input().split()))
b=set(map(int, input().split()))
for i in n_array:
    if i in a:
        happiness +=1
    elif i in b:
        happiness -=1
    elif i not in a or b:
        happiness= happiness

print(happiness)

#Check Strict Superset 13
setA = set(input().split())
N= int(input()) 
count = 0

for i in range(N):
    a=set(input().split())
    if setA.issuperset(a):
       count += 1
print(count == N)


###### NUMPY ######

#Arrays 1
import numpy 
def arrays(arr):
    arr=list(arr)
    arr.reverse()
    arr2=[i for i in arr]
    result= numpy.array(arr2, float)
    return result

#Shape and Reshape 2
import numpy

array_2= input().split(" ")
f= [int(i) for i in array_2]
print(numpy.reshape(f,(3,3)))

#Transpose and Flatten 3
import numpy

dim= input().strip().split(" ")
row= int(dim[0])
col= int(dim[1])
matrix = []
for i in range(row):
    row=list(map(int, input().split()))
    matrix.append(row)
m= [x for x in matrix]
m= numpy.array(m)
print(numpy.transpose(m))
print(m.flatten())

#Min and Max 4 
import numpy

dim= input().strip().split(" ")
row= int(dim[0])
col= int(dim[1])
matrix = []
for i in range(row):
    row=list(map(int, input().split()))
    matrix.append(row)
m= [x for x in matrix]
m= numpy.array(m)
minimum= numpy.min(m, axis=1)
print(numpy.max(minimum))

#Mean, Var, and Std 5
import numpy

dim= input().strip().split(" ")
row= int(dim[0])
col= int(dim[1])
matrix = []
for i in range(row):
    row=list(map(int, input().split()))
    matrix.append(row)
m= [x for x in matrix]
m= numpy.array(m)
print(numpy.mean(m, axis = 1))
print(numpy.var(m, axis=0))
print(numpy.around(numpy.std(m, axis = None),11))

#Zeros and Ones 6
import numpy
dim= list(map(int,input().strip().split()))
print(numpy.zeros((dim), dtype = numpy.int))
print(numpy.ones((dim), dtype= numpy.int))

#Eye and Identity 7
import numpy as np

A=tuple(map(int, input().split()))
np.set_printoptions(legacy= '1.13')
print(np.eye(A[0], A[1], k= 0 ))

#Array Mathematics 8
import numpy as np
A=input().strip().split(" ")
N=int(A[0])

arrayA= np.array([list(map(int,input().split(" "))) for x in range (N)])
arrayB= np.array([list(map(int,input().split(" "))) for x in range (N)])

print(arrayA + arrayB)
print(np.subtract(arrayA, arrayB))
print(np.multiply(arrayA, arrayB))
print(np.floor_divide(arrayA, arrayB))
print(np.mod(arrayA, arrayB))
print(np.power(arrayA, arrayB))

#Floor, Ceil and Rint 9
import numpy as np
arrayA= np.array(input().split(" "), float)
np.set_printoptions(legacy='1.13')
print(np.floor(arrayA))
print(np.ceil(arrayA))
print(np.rint(arrayA))

#Sum and Prod 10
import numpy as np


dim= input().strip().split(" ")
row= int(dim[0])
col= int(dim[1])
matrix = []
for i in range(row):
    row=list(map(int, input().split()))
    matrix.append(row)
m= [x for x in matrix]
m= np.array(m)

summ= np.sum(m, axis = 0)
print(np.prod(summ, axis = None)) 

# Dot and Cross 11
import numpy as np

N= int(input())
matrixA= []
matrixB= []
for i in range(N):
    matrixA.append(list(map(int, input().split())))
for x in range(N):
    matrixB.append(list(map(int, input().split())))

m_A=np.array(matrixA)
m_B=np.array(matrixB)
 
print(np.dot(m_A,m_B))

#Inner and Outer 12
import numpy

a= list(map(int, input().split()))
b= list(map(int, input().split()))
print(numpy.inner(a,b))
print(numpy.outer(a,b))

#Polynomials 13 
import numpy
Poly= list(map(float, input().split()))
x= int(input())
print(numpy.polyval(Poly, x))

#Linear Algebra 14
import numpy as np

N=int(input())
matrixA=[]
for i in range(N):
    r=list(map(float, input().split()))
    matrixA.append(r)
A= np.array(matrixA)
print(round(np.linalg.det(A),2))

#Concatenate 15
import numpy 
N= input().split(" ")
n= int(N[0])
m= int(N[1])
p= int(N[2])

A=numpy.array([input().split() for i in range(n)], int)
B=numpy.array([input().split() for i in range(m)], int)

print(numpy.concatenate((A, B), axis= 0))


#### BUILT IN ####

#Input() 1
x, k=map(int,input().split(" "))
P= input()
print(eval(P) == k)

# Python Evaluation 2
eval(input())

#Any or All 3
N=int(input())
lista= list(map(int,input().split()))
all_1=all(x>=0 for x in lista)
any_1=any(str(x)== str(x)[::-1] for x in lista)
print(all_1 and any_1)

##### COLLECTIONS #####

#collections.Counter() 1
from collections import Counter 

shoes= int(input())
sizes= Counter(list(map(int, input().split(" "))))
c=(int(input()))
money=0
for i in range(c):
    size_price= list(map(int, input().split()))
    if sizes[size_price[0]] != 0:
        money += size_price[1]
        sizes[size_price[0]] -= 1
        
print(money)

#Collections.deque() 2
from collections import deque
N= int(input())
d=deque()
for i in range(N):
    lista=list(input().split(" "))
    if lista[0] == "append":
        d.append(lista[1])
    if lista[0] == "appendleft":
        d.appendleft(lista[1])
    if lista[0]== "pop":
        d.pop()
    if lista[0]== "popleft":
        d.popleft()
        
print(" ".join(str(x) for x in d))

# Collections.OrderedDict() 3
from collections import OrderedDict
N= int(input())
market=OrderedDict()
for i in range(N):
    lista=input().split(" ")
    item= " ".join(lista[:-1])
    price=int(lista[-1])
    if item in market:
        market[item] += price
    else:
        market[item] = price
        

for key, value in market.items():
    print(key, value)

#  Collections.namedtuple() 4
from collections import namedtuple

n= int(input())
students= namedtuple("students", input().split())
m= []

for i in range(n):
    r= input().split()
    m.append(students(*r))
average= sum(int(x.MARKS) for x in m)
print(f"{average / n:.2f}")

#Company Logo 5
s=input()
s= sorted(s)
tot=Counter(s)
for i, t in tot.most_common(3):
    print("%s %d" % (i, t))

#Word Order 6
from collections import Counter
from collections import OrderedDict

n=int(input())
list1= list()
for i in range(0,n):
    list1.append(input())

print(len(list(OrderedDict.fromkeys(list1))))
    
c=Counter(list1)
u=list(c.values())
for each in u:
    print(each, end= " ")
print()

#Piling Up! 7
for i in range(int(input())):
    n=int(input())
    c= list(map(int, input().split()))
    m=min(c)
    mi = c.index(m)
    left= c[:mi]
    right= c[mi+1:]
    if left==sorted(left, reverse= True) and right== sorted(right):
        print ('Yes')
    else: 
        print('No')

#DefaultDict Tutorial 8
from collections import defaultdict
n, m= map(int,input().split())
d= defaultdict(list)
for i in range(n):
    a= input()
    d[a].append(i+1)
for j in range(m):
    b = input()
    if b in d:
        print(*d[b])
    else:
        print(-1)





##### DATE AND TIME #####

#Calendar Module 1
import calendar
date_1= list(map(int, input().split(" ")))
cale= {0: 'MONDAY', 1: 'TUESDAY', 2: 'WEDNESDAY', 3: 'THURSDAY', 4: 'FRIDAY', 5: 'SATURDAY', 6: 'SUNDAY'}
if 2000<(date_1[2])< 3000:
    number= calendar.weekday(date_1[2], date_1[0], date_1[1])
    print(cale[number])

#Time Delta 2
def time_delta(t1, t2):
    t1= datetime.strptime(t1, "%a %d %b %Y %H:%M:%S %z")
    t2= datetime.strptime(t2, "%a %d %b %Y %H:%M:%S %z")
    result= int(abs(t1-t2).total_seconds())
    return str(result)





##### ERRORS AND EXCEPTIONS #####

#Exceptions 1
n= int(input())
for i in range(n):
    a= list(input().split())
    try:
        print(int(a[0])//int(a[1]))
    except (ZeroDivisionError, ValueError) as e:
        print ("Error Code:", e)

#### PYTHON FUNCTIONALS ####

#Reduce Function 
def product(fracs):
    t = reduce(lambda a, b: a*b, fracs)
    return t.numerator, t.denominator

### CLOSURES AND DECORATORS ###

#Standardize Mobile Number Using Decorators 1
from re import sub
def wrapper(f):
    a= r"^(?:\+?91|0)??(\d{5})(\d{5})$"
    b= r"+91 \1 \2"
    
    def fun(l):
        return f([sub(a, b, i) for i in l])
    return fun

#Decorators 2 - Name Directory 2
def person_lister(f):
    def inner(people):
        people.sort(key=lambda x: int(x[2]))
        for i in people:
            yield f(i)
    return inner

#### REGEX AND PARSING #####

#Re.split() 1
regex_pattern = r"[.,]"

#Validating phone numbers 2
import re

for i in range(int(input())):
    string= input()
    if len(string)==10 and string.isdigit() == True:
        match= re.search(r"^[987]", string)
        if match:
            print("YES")
        else:
            print("NO")
    else:
        print("NO")

#Detect Floating Point Number 3
import re

pattern=r"^[+-]?\d*\.\d+$"
for i in range(int(input())):
    s=input()
    if re.match(pattern, s):
        print("True")
    else:
        print("False")

#Validating Roman Numerals 4
regex_pattern = r"^(?!.*(I{4}|V{2,}|X{4}|L{2,}|C{4}|D{2,}|M{4})).*$"

#Group(), Groups() & Groupdict() 5
import re 
s= input()
m= re.search(r'([a-zA-Z0-9])\1+', s)
if m:
    print(m.group(1))
else:
    print('-1')

#Re.findall() & Re.finditer() 6
import re
s =input()
r =re.findall(r'(?<=[^aeiouAEIOU])[aeiouAEIOU]{2,}(?=[^aeiouAEIOU])',s)
if not r:
    print(-1)
else:
    for i in r:
        print(i)

#Re.start() & Re.end() 7
import re
s= input()
k= f"(?={input()})"
length = len(k) - 4
for m in re.finditer(k, s):
    print(f"({m.start()}, {m.start() + length -1})")
if not re.search(k, s):
    print("(-1, -1)")

#Hex Color Code 8
import re
pattern= r":?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})"
for i in range(int(input())):
    s= input()
    matches= re.findall(pattern, s)
    if matches:
        print(*matches, sep= "\n")


#Validating UID 9
import re
pattern= r'^(?=(.*[A-Z]){2})(?=(.*\d){3})(?!.*(.).*\3).{10}$'
for i in range(int(input())):
    m=re.match(pattern, input())
    if m:
        print('Valid')
    else:
        print('Invalid')
    

#Validating Credit Card Numbers 10
import re

pat = re.compile(r"^(?!.*(\d)(-?\1){3})([4-6]\d{3})(-?)(\d{4})(-?)(\d{4})(-?)(\d{4})$")

for x in range(int(input())):
    c = input().strip()
    if pat.match(c):
        print("Valid")
    else:
        print("Invalid")

#Validating Postal Codes 11
regex_integer_in_range = r"^[100000-999999]{6}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)" 	# Do not delete 'r'.

#Validating and Parsing Email Addresses 12
import re 
pattern = r"^[a-zA-Z][\w.-]+@[a-zA-Z]+\.[a-z]{1,3}$"
for i in range(int(input())):
    name,email = input().split()
    email= email[1:-1]
    if re.match(pattern, email):
        print(f"{name} <{email}>")

#HTML Parser - Part 1  13 
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('Start :', tag)
        self.handle_attrs(attrs)

    def handle_endtag(self, tag):
        print('End   :', tag)

    def handle_startendtag(self, tag, attrs):
        print('Empty :', tag)
        self.handle_attrs(attrs)

    def handle_attrs(self, attrs):
        for attribute in attrs:
            print('->', attribute[0], '>', attribute[1])


parser = MyHTMLParser()

n = int(input())
for _ in range(n):
    html = input()
    parser.feed(html)

#Detect HTML Tags, Attributes and Attribute Values 14
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attribute in attrs:
            print('-> {} > {}'.format(*attribute))


html = '\n'.join([input() for _ in range(int(input()))])
parser = MyHTMLParser()
parser.feed(html)
parser.close()

#HTML Parser - Part 2    15
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        number_of_line = len(data.split('\n'))
        if number_of_line > 1:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        if data.strip():
            print(data)

    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)


parser = MyHTMLParser()

n = int(input())
html_string = ''
for i in range(n):
    html_string += input().rstrip() + '\n'

parser.feed(html_string)

#Matrix Script   16
n, m = map(int, input().split())
data = []
for i in range(n):
    data.append(input())

string = ''
for x in zip(*data):
    string += ''.join(x)

print(re.sub(r'(?<=\w)([^\w]+)(?=\w)', ' ', string))


#Regex Substitution 17
import re

for i in range(int(input())):
    string = input()
    newStr= re.sub(r'(?<= )&&(?= )', 'and', string) 
    newStr= re.sub(r'(?<= )\|\|(?= )', "or", newStr)
    print(newStr)


#### XML #####

#XML 1 - Find the Score 
def get_attr_number(node):
    a= len(node.attrib)
    if len(node) == 0:
        return a
    return a + sum(get_attr_number(i) for i in node)

#XML2 - Find the Maximum Depth
maxdepth = 0
def depth(elem, level):
    global maxdepth
    level = level + 1
    for e in elem:
        depth(e, level)
    if(level > maxdepth):
        maxdepth = level



###Birthday Cake Candles
def birthdayCakeCandles(candles):
    m=max(candles)
    count= 0
    for i in candles:
        if i == m:
            count +=1
    return count

###Insertion Sort - Part 1
def insertionSort1(n, arr):
    a= arr[-1]
    i= n-1
    while i >0 and arr[i-1] > a:
        arr[i]=arr[i-1]
        print(*arr)
        i-=1
    arr[i]=a
    print(*arr)

###Insertion Sort - Part 2
def insertionSort2(n, arr):
    for i in range(1,n):
        a= arr[i]
        t=i      
        while t>0 and arr[t-1]>a:
            arr[t]=arr[t-1]
            t-=1
        arr[t]=a
        print(*arr)

###Recursive Digit Sum
def summation(final_numb):
    return sum(int(x) for x in list(final_numb))
    
def recursive_digit_sum(num):
    if num < 10:
        return num
    else:
        next_sum = summation(str(num))
        return recursive_digit_sum(next_sum)
        
def superDigit(n, k):
    initial_sum= summation(n) * k
    return recursive_digit_sum(initial_sum)


####Viral Advertising 
def viralAdvertising(n):
    shared =5
    cumulative=0
    for i in range(1,n+1):
        liked = shared//2
        cumulative+=liked
        shared = liked*3
    return cumulative

####Number Line Jumps
def kangaroo(x1, v1, x2, v2):
    if v1 != v2:
        if (x2 - x1) % (v1 - v2) == 0 and (x2 - x1) / (v1 - v2) > 0:
            return "YES"
        else:
            return "NO"
    else:
        if x1 == x2:
            return "YES"
        else:
            return "NO"