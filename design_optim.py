from GA import GA_Design_Optim


GA_optimizer = GA_Design_Optim(decode_size = 10,)

GA_optimizer.evolve()
GA_optimizer.save_fig()