import tensorflow as tf

BATCH_SIZE = 64 # @param
NUM_LATENTS = 20 # @param
TRAINING_STEPS = 10000 # @param

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_images = mnist.train.images.reshape((-1, 28, 28, 1))

dataset = tf.data.Dataset.from_tensor_slices(train_images)
batched_dataset = dataset.shuffle(100000).repeat().batch(BATCH_SIZE)

iterator = batched_dataset.make_one_shot_iterator()
images = iterator.get_next()

# Rescale data
real_data = 2 * images - 1

latents = tf.random_normal((BATCH_SIZE, NUM_LATENTS))
generator = MnistGenerator()
samples = generator(latents)

discriminator = MnistDiscriminator()

discriminator_real_data_logits = discriminator(real_data)
discriminator_real_data_labels = tf.ones(shape=BATCH_SIZE, dtype=tf.int32)

real_data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=discriminator_real_data_logits, labels=discriminator_real_data_labels)

discriminator_samples_logits = discriminator(samples)
discriminator_samples_labels = tf.zeros(shape=BATCH_SIZE, dtype=tf.int32)
samples_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=discriminator_samples_logits, labels=discriminator_samples_labels)

# Reduce loss over batch dimension
discriminator_loss = tf.reduce_mean(real_data_loss + samples_loss)

# Ignore for the first exercise, uncomment if you do the last exercise.
# discriminator_loss += 10 * dragan_penalty_loss(discriminator, real_data)



# Get the probabilities from the discriminator logits.
discriminator_probabilities = tf.nn.softmax(discriminator_samples_logits)
# We have to index the discrimiantor output to obtain the probability that the 
# samples are fake.
generator_loss = - tf.log(discriminator_probabilities[:, 1])
generator_loss = tf.reduce_mean(generator_loss)


discrimiantor_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5, beta2=0.9)
generator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5, beta2=0.9)

# Optimize the discrimiantor.
discriminator_update_op = discrimiantor_optimizer.minimize(
    discriminator_loss, var_list=discriminator.get_all_variables())

# Optimize the generator..
generator_update_op = generator_optimizer.minimize(
    generator_loss, var_list=generator.get_all_variables())


sess = tf.Session()

# Initialize all variables
sess.run(tf.global_variables_initializer())

disc_losses = []
gen_losses = []

for i in xrange(TRAINING_STEPS):
  sess.run(discriminator_update_op)
  sess.run(generator_update_op)

  if i % 100 == 0: 
    disc_loss = sess.run(discriminator_loss)
    gen_loss = sess.run(generator_loss)

    disc_losses.append(disc_loss)
    gen_losses.append(gen_loss)

    print('At iteration {} out of {}'.format(i, TRAINING_STEPS))

