---
layout: post
title: "Turtle in Python: A Traffic light"
date: 2017-08-11
categories: [Technical Fridays, Python]
---

We’re going to build a program that uses a turtle in python to simulate the traffic lights.

There will be four states in our traffic light: Green, then Green and Orange together, then Orange only, and then Red.
The light should spend 3 seconds in the Green state, followed by one second in the Green+Orange state, then one second in the Orange state, and then 2 seconds in the Red state.


{% highlight python %}
import turtle  # Allows us to use turtles

turtle.setup(400, 600)  # Determine the window size
wn = turtle.Screen()  # Creates a playground for turtles
wn.title('traffic light using different turtles')  # Set the window title
wn.bgcolor('skyblue')  # Set the window background color
tess = turtle.Turtle()  # Create a turtle, assign to tess
alex = turtle.Turtle()  # Create alex
henry = turtle.Turtle()  # Create henry


def draw_housing():
    """ Draw a nice housing to hold the traffic lights"""
    tess.pensize(3)  # Change tess' pen width
    tess.color('black', 'white')  # Set tess' color
    tess.begin_fill()  # Tell tess to start filling the color
    tess.forward(80)  # Tell tess to move forward by 80 units
    tess.left(90)  # Tell tess to turn left by 90 degrees
    tess.forward(200)
    tess.circle(40, 180)  # Tell tess to draw a semi-circle
    tess.forward(200)
    tess.left(90)
    tess.end_fill()  # Tell tess to stop filling the color


draw_housing()


def circle(t, ht, colr):
    """Position turtle onto the place where the lights should be, and
    turn turtle into a big circle"""
    t.penup()  # This allows us to move a turtle without drawing a line
    t.forward(40)
    t.left(90)
    t.forward(ht)
    t.shape('circle')  # Set tutle's shape to circle
    t.shapesize(3)  # Set size of circle
    t.fillcolor(colr)  # Fill color in circle


circle(tess, 50, 'green')
circle(alex, 120, 'orange')
circle(henry, 190, 'red')

{% endhighlight %}

We're going to use the concept of *state machine*.
A state machine is a system that can be in one of a few different states.  
This idea is not new: for example, when first turning on a cellphone, it goes into a state which we could call “Awaiting PIN”. When the correct PIN is entered, it transitions into a different state — say “Ready”. Then we could lock the phone, and it would enter a “Locked” state, and so on.  

A traffic light is a kind of state machine with four states: Green, then Green+Orange, then Orange only, and then Red.
We number these states 0, 1, 2 and 3. When the machine changes state, we change turtle’s position and its color.

{% highlight python %}
# This variable holds the current state of the machine
state_num = 0


def advance_state_machine():
    """A state machine for traffic light"""
    global state_num  # Tells Python not to create a new local variable for state_num

    if state_num == 0:  # Transition from state 0 to state 1
        henry.color('darkgrey')
        alex.color('darkgrey')
        tess.color('green')
        wn.ontimer(advance_state_machine, 3000)  # set the timer to explode in 3 sec
        state_num = 1
    elif state_num == 1:  # Transition from state 1 to state 2
        henry.color('darkgrey')
        alex.color('orange')
        wn.ontimer(advance_state_machine, 1000)
        state_num = 2
    elif state_num == 2:  # Transition from state 2 to state 3
        tess.color('darkgrey')
        wn.ontimer(advance_state_machine, 1000)
        state_num = 3
    else:                 # Transition from state 3 to state 0
        henry.color('red')
        alex.color('darkgrey')
        wn.ontimer(advance_state_machine, 2000)
        state_num = 0


advance_state_machine()
{% endhighlight %}

Now we need to tell the window to start listening for events.
{% highlight python %}
wn.listen()  # Listen for events

wn.mainloop()  # Wait for user to close window
{% endhighlight %}

Our traffic light will look like this:

<div style="text-align: center">
<video controls width="600" height="337"><source src="/assets/traffic_light.webm" type="video/webm">Your browser doesn't support video tag or WebM!</video>
</div>

Below is the same program in <abbr title="Works best on desktop site">interactive mode</abbr> with minor modifications:

<div id="tk">
{% include trinket-open type='python' %}
import turtle

# Create a playground for turtles
wn = turtle.Screen()
wn.bgcolor('skyblue')

# Create turtles
tess = turtle.Turtle()
alex = turtle.Turtle()
henry = turtle.Turtle()


def draw_housing():
    """ Draw a nice housing to hold the traffic lights"""
    tess.pensize(3)
    tess.color('black', 'white')
    tess.begin_fill()
    tess.forward(80)
    tess.left(90)
    tess.forward(157)
    tess.circle(40, 180)
    tess.forward(157)
    tess.left(90)
    tess.end_fill()


draw_housing()


def circle(t, ht, colr):
    """Position turtle onto the place where the lights should be, and
    turn turtle into a big circle"""
    t.penup()
    t.forward(40)
    t.left(90)
    t.forward(ht)
    t.shape('circle')
    t.fillcolor(colr)


circle(tess, 40, 'green')
circle(alex, 100, 'orange')
circle(henry, 160, 'red')

# This variable holds the current state of the machine
state_num = 0


def advance_state_machine():
    global state_num  # The global keyword tells Python not to create a new local variable for state_num

    if state_num == 0:  # Transition from state 0 to state 1
        henry.color('darkgrey')
        alex.color('darkgrey')
        tess.color('green')
        wn.ontimer(advance_state_machine, 3000)  # set the timer to explode in 3000 milliseconds (3 seconds)
        state_num = 1
    elif state_num == 1:  # Transition from state 1 to state 2
        henry.color('darkgrey')
        alex.color('orange')
        wn.ontimer(advance_state_machine, 1000)
        state_num = 2
    elif state_num == 2:  # Transition from state 2 to state 3
        tess.color('darkgrey')
        wn.ontimer(advance_state_machine, 1000)
        state_num = 3
    else:                 # Transition from state 3 to state 0
        henry.color('red')
        alex.color('darkgrey')
        wn.ontimer(advance_state_machine, 2000)
        state_num = 0


advance_state_machine()

wn.listen()  # Listen for events

wn.mainloop()  # Wait for user to close window

{% include trinket-close %}
</div>

**Resources:**  
The above program is an exercise from the book [Think Python](http://www.greenteapress.com/thinkpython/html/index.html).
