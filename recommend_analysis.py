from data_processing import get_data

data = get_data()

# Slice the title to 50 characters
# This number of characters is enough to identify the product
data['product/title'] = data['product/title'].str.slice(0, 50)
number_of_reviews = data['product/title'].value_counts()
product_scores_avg = data.groupby(['product/title'])['review/score'].mean()

# Normalize data so the scores are in range [0, 10]
normalized_scores = (product_scores_avg - product_scores_avg.min()) / (product_scores_avg.max() - product_scores_avg.min()) * 10

# Print top 10 products with the highest average score and more than 10 reviews
print(normalized_scores[number_of_reviews > 10].sort_values(ascending=False).head(10))

# Print top 10 products with the lowest average score and more than 10 reviews
print(normalized_scores[number_of_reviews > 10].sort_values(ascending=True).head(10)[::-1])

import matplotlib.pyplot as plt

# Get the best products, with more than 10 reviews
best_products = normalized_scores[number_of_reviews > 10].sort_values(ascending=False)

# Get the worst products, with more than 10 reviews
worst_products = normalized_scores[number_of_reviews > 10].sort_values(ascending=True)

# Plot the best products
fig, ax = plt.subplots(figsize=(16, 12))
# Horizontal Bar Plot
ax.barh(best_products.head(10).index, normalized_scores[best_products.head(10).index])

# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5, labelsize=12)
ax.yaxis.set_tick_params(pad=10, labelsize=12)

# Add x, y gridlines
ax.grid(color='grey',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)

# Show top values
ax.invert_yaxis()

# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width() + 0.01, i.get_y() + 0.5,
             str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold',
             color='grey')

# Add Plot Title
ax.set_title('Best products',
             loc='center', fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel('Score', fontsize = 16)
ax.set_ylabel('Product name', fontsize=16)

plt.subplots_adjust(left=0.35)

# Show Plot
plt.savefig('best_products_chart.png')
plt.show()



# Plot the worst products
fig, ax = plt.subplots(figsize=(16, 12))
# Horizontal Bar Plot
ax.barh(worst_products.head(10).index, normalized_scores[worst_products.head(10).index])

# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5, labelsize=12)
ax.yaxis.set_tick_params(pad=10, labelsize=12)

# Add x, y gridlines
ax.grid(color='grey',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)

# Show top values
ax.invert_yaxis()

# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width() + 0.01, i.get_y() + 0.5,
             str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold',
             color='grey')

# Add Plot Title
ax.set_title('Worst products',
             loc='center', fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel('Score', fontsize = 16)
ax.set_ylabel('Product name', fontsize=16)

plt.subplots_adjust(left=0.35)

# Show Plot
plt.savefig('worst_products_chart.png')
plt.show()

# Plot the distribution of the scores for the 1000 best products

plt.figure(figsize=(10, 5))
plt.title('Distribution of the scores for the best products')
plt.xlabel('Score')
plt.ylabel('Number of products')
plt.hist(normalized_scores[best_products.head(100).index], bins=10)
plt.savefig('1000_best_scores.png')
plt.show()

# Plot the distribution of the scores for the 1000 worst products

plt.figure(figsize=(10, 5))
plt.title('Distribution of the scores for the worst products')
plt.xlabel('Score')
plt.ylabel('Number of products')
plt.hist(normalized_scores[worst_products.head(1000).index], bins=10)
plt.savefig('1000_worst_scores.png')
plt.show()

# Print number of unique products for the 'product/productId' column
print('Number of unique products: ', data['product/productId'].nunique())

# Number of unique products: 7438

"""
In this section, after looking the plots, we can see that in the worst 1000 products,
the scores are mostly < 8.0, and in the best 1000 products, the scores are mostly > 8.0. 

This means, that in the recommendary system, we can use the scores as a feature, and recommend
only products, that have a score >= 8.0. This will increase the quality of the recommendations,
and provide good products to the users.

Also, I would like to mention, that unique number of products is almost 7500, so if we decide to recommend
products with a score > 7.5, we will have a lot of products to recommend, still maintaining high quality products.
"""


