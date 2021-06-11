import random
from deap import base, creator, tools

#определим функцию оценки 
def eval_func(individual):

    target_sum = 15
    return len(individual) - abs(sum(individual) - target_sum),

#создаем набор инструментов с правильными параметрами
def create_toolbox(num_bits):

    creator.create("FitnessMax", base.Fitness, weights = (1.0,))
    creator.create("Individual", list, fitness = creator.FitnessMax)

    #инициализируем панель инстрментов 
    toolbox = base.Toolbox()    
    toolbox.register("attr_bool", random.randint, 0, 1)#генерируем атрибуты   
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_bits)#инициализируем структуры    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)#определяем популяцию как список людей    
    toolbox.register("evaluate", eval_func)#оператор оценки    
    toolbox.register("mate", tools.cxTwoPoint)#Оператор скрещивания    
    toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)#Оператор мутации    
    toolbox.register("select", tools.selTournament, tournsize = 3)#Оператор для разведения

    return toolbox

if __name__ == "__main__":   
    
    num_bits = 45#определеяем количество бит
    toolbox = create_toolbox(num_bits)#создаем набор инструментов, используя num_bits
    random.seed(7)#генератор случайных чисел

    population = toolbox.population(n = 500)#создаем начальную популяцию в 500 поколений
    probab_crossing, probab_mutating = 0.5, 0.2 #определяем вероятности скрещивания и мутации
    num_generations = 10 #определяем количество поколений
    print('\nEvolution process starts')

    #Оценка всего населения
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    print('\nEvaluated', len(population), 'individuals')

    #Создадим и выведем на печать поколения
    for g in range(num_generations):
        print("\n --- Generation", g)        
        offspring = toolbox.select(population, len(population))#Выбор следующего поколения        
        offspring = list(map(toolbox.clone, offspring))#клонируем выбранные поколения

        #Применяем скрещиваение и мутацию на потомство
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            
            #скрещиваем два поколения
            if random.random() < probab_crossing:
                toolbox.mate(child1, child2)

                #Удаляем значение фитнеса ребенка
                del child1.fitness.values
                del child2.fitness.values

        #применим мутацию
        for mutant in offspring:

            #мутируем поколение
            if random.random() < probab_mutating:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        #Оценим особи с недопустимой пригодностью
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print('Evaluated', len(invalid_ind), 'individuals')

        #заменим население на следующее поколение
        population[:] = offspring

        #печатаем статистику по текущим поколениям
        fits = [ind.fitness.values[0] for ind in population]

        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print('Min =', min(fits), ', Max =', max(fits))
        print('Average =', round(mean, 2), ', Standard deviation =', round(std, 2))

    print("\n --- Evolution ends")

    best_ind = tools.selBest(population, 1)[0]
    print('\nBest individual:\n', best_ind)
    print('\nNumber of ones:', sum(best_ind))