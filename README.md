Using a linear regression model that was coded from scratch, predicted the salary based on experience.
The dataset has only one independant feature (years of experience) and one target (salary)

Workflow of Linear Regression

1. Initialise learning rate, no of iterations, weights (numpy array) and bias (constant)
2. Build Linear Regression equation (y = w * x + b)
3. Find the predicted y value for each x value for the corresponding weight and bias
4. Check the loss function for these parameter values
5. Update the  value for weight and bias using Gradient Descent
6. Steps 3, 4, 5 are repeated till the value  of the loss function becomes  minimum
