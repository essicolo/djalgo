import copy
import random
from .analysis import Index

class Darwin:
    def __init__(
            self, initial_phrases, mutation_rate=0.05, population_size=50,
            mutation_probabilities=None,
            scale=None, measure_length = 4, time_resolution=[0.125, 4],
            weights = None, targets=None, seed=None
        ):
        """
        Initialize the Darwin class for evolving musical phrases.

        Args:
            initial_phrases (list): List of initial musical phrases, each a list of tuples (pitch, duration, offset).
            index_weights (dict): Weights for each musical aspect ('gini', 'balance', 'motif', 'dissonance').
            mutation_rate (float): Probability of mutation per attribute in a phrase.
            population_size (int): Number of phrases in each generation.
            scale (list): List of pitches considered harmonious or in-scale; used for dissonance calculation.
        """
        if mutation_probabilities:
            self.mutation_probabilities = mutation_probabilities
        else:
            self.mutation_probabilities = {
                'pitch': lambda: random.choice(self.scale),
                'duration': lambda: random.choice(self.possible_durations),
                'rest': lambda: None if random.random() < 0.02 else 1  # 2% chance of a rest
            }
        self.population = [self.mutate(phrase, rate=0) for phrase in initial_phrases for _ in range(population_size // len(initial_phrases))]
        self.weights = weights
        self.mutation_rate = mutation_rate
        self.population_size = population_size

        self.scale = scale
        self.measure_length = measure_length
        self.time_resolution = time_resolution
        if self.time_resolution[1] > self.measure_length:
            print("Warning: Max duration exceeds measure length. Decreasing max duration to measure length.")
            self.time_resolution[1] = self.measure_length

        all_durations = [0.125, 0.25, 0.5, 1, 2, 3, 4, 8]
        self.possible_durations = [d for d in all_durations if self.time_resolution[0] <= d <= self.time_resolution[1]]
        # Set the weights and targets for each metric are set to default values if not provided
        if weights is None:
            self.weights = {
            'gini': (1.0, 1.0, 0.0),
            'balance': (1.0, 1.0, 0.0),
            'motif': (10.0, 1.0, 0.0),
            'dissonance': (1.0, 0.0, 0.0),
            'rhythmic': (0, 10.0, 0),
            'rest': (1.0, 0.0, 0.0)
        }
        else:
            self.weights = weights
        if targets is None:
            self.targets = {
            'gini': (0.05, 0.5, 0.0),  # Example targets; adjust as needed
            'balance': (0.1, 0.1, 0.0),
            'motif': (1.0, 1.0, 0.0),
            'dissonance': (0.0, 0.0, 0.0),
            'rhythmic': (0.0, 1.0, 0.0),  # Assuming rhythmic applies to duration
            'rest': (0.0, 0.0, 0.0) # rests are None pitches
        }
        else:
            self.targets = targets
        
        if seed is not None:
            random.seed(seed)

        self.best_individuals = []  # List to store the best individual of each generation
        self.best_scores = []
    
    def calculate_fitness_components(self, phrase):
        """
        Calculate the fitness components based on the Index class for a given phrase.

        Args:
            phrase (list): The musical phrase.

        Returns:
            dict: A dictionary containing the fitness components ('gini', 'balance', 'motif', 'dissonance', 'rhythmic').
        """
        fitness_components = {}
        # Split the phrase into its components
        pitches, durations, offsets = zip(*phrase)

        # Gini coefficient for pitch and duration
        fitness_components['gini_pitch'] = Index(pitches).gini()
        fitness_components['gini_duration'] = Index(durations).gini()

        # Balance for pitch and duration
        fitness_components['balance_pitch'] = Index(pitches).balance()
        fitness_components['balance_duration'] = Index(durations).balance()

        # Motif for pitch and duration (and potentially offsets if applicable)
        fitness_components['motif_pitch'] = Index(pitches).motif()
        fitness_components['motif_duration'] = Index(durations).motif()  # Add this if you're analyzing motifs in durations too

        # Dissonance, only applicable to pitches
        if self.scale is not None:
            fitness_components['dissonance_pitch'] = Index(pitches).dissonance(self.scale)

        # Rhythmic fit based on durations
        # Note: Here, I am assuming you have a method in Index class for rhythm analysis. Adjust if necessary.
        fitness_components['rhythmic'] = Index(durations).rhythmic(self.measure_length)  # Adjust method as needed

        # gini and balance for offsets?
        #fitness_components['gini_offset'] = Index(offsets).gini()  # Add if relevant
        #fitness_components['balance_offset'] = Index(offsets).balance()  # Add if relevant

        # Calculate the proportion of rest notes (notes where pitch is None)
        rest_notes = [note for note in phrase if note[0] is None]
        total_notes = len(phrase)
        rest_proportion = len(rest_notes) / total_notes if total_notes > 0 else 0
        fitness_components['rest'] = rest_proportion

        return fitness_components


    def fitness(self, phrase):
        """
        Calculate the fitness of a phrase based on how closely it meets the desired musical metrics targets.

        Args:
            phrase (list): Musical phrase.

        Returns:
            float: Fitness score of the phrase.
        """
        fitness_score = 0.0
        components = self.calculate_fitness_components(phrase)

        # Iterate through each metric and calculate the contribution to fitness
        for metric, (pitch_target, duration_target, offset_target) in self.targets.items():
            for i, component in enumerate(['pitch', 'duration', 'offset']):
                actual_value = components.get(f"{metric}_{component}", 0)
                target_value = (pitch_target, duration_target, offset_target)[i]
                weight = self.weights[metric][i]

                # Calculate the similarity from the target (only if there is a target)
                if target_value is not None and weight > 0:  # Ensure there's a target and weight for this component
                    similarity = 1 - abs(actual_value - target_value) / (target_value if target_value != 0 else 1)
                    fitness_score += max(0, similarity) * weight  # Apply weight and ensure non-negative score

        return fitness_score

    def mutate(self, phrase, rate=None, rest_rate=0.02):
        """
        Mutate a musical phrase while respecting musical structures and boundaries.
        """
        if rate is None:
            rate = self.mutation_rate

        new_phrase = []
        total_offset = 0  # Track total offset for alignment

        for note in phrase:
            pitch, duration, offset = note

            # Mutate pitch, duration, and rest based on mutation probabilities
            new_pitch = self.mutation_probabilities['pitch']() if random.random() < rate else pitch
            new_duration = self.mutation_probabilities['duration']() if random.random() < rate else duration
            new_rest = self.mutation_probabilities['rest']()

            # Align offsets based on sequential order, ensuring notes/rests align with previous ones
            new_offset = total_offset
            total_offset += new_duration  # Increment total offset by new duration

            # Append the new note or rest to the phrase
            new_phrase.append((new_pitch if new_rest else None, new_duration, new_offset))

        return new_phrase

    def select(self, k=25):
        """
        Select top-k phrases based on fitness.

        Args:
            k (int): Number of top phrases to select.

        Returns:
            list: Top k musical phrases.
        """
        # Evaluate the fitness of each phrase in the population
        fitness_scores = [(phrase, self.fitness(phrase)) for phrase in self.population]
        # Sort the phrases by their fitness scores in descending order
        sorted_phrases = sorted(fitness_scores, key=lambda x: x[1], reverse=True)
        # Select the top-k phrases
        selected_phrases = [phrase for phrase, score in sorted_phrases[:k]]
        return selected_phrases

    def crossover(self, parent1, parent2):
        """
        Combine two phrases to create a new phrase.

        Args:
            parent1, parent2 (list): Parent phrases.

        Returns:
            list: New musical phrase generated from parents.
        """
        # Cut points for the crossover
        cut1, cut2 = sorted(random.sample(range(len(parent1)), 2))
        # Create new phrase by combining parts of the parent phrases
        new_phrase = parent1[:cut1] + parent2[cut1:cut2] + parent1[cut2:]
        return new_phrase

    def evolve(self, k=25, rest_rate=0.02):
        """
        Evolve the population of phrases through selection, crossover, and mutation.

        Args:
            k (int): Number of phrases to select for the next generation.
        """
        # Selection based on fitness
        selected_phrases = self.select(k)

        # Store the best individual and its fitness score
        self.best_individuals.append(selected_phrases[0])
        self.best_scores.append(self.fitness(selected_phrases[0]))

        # Crossover and mutation to create a new generation
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(selected_phrases, 2)
            parent1 = copy.deepcopy(parent1)
            parent2 = copy.deepcopy(parent2)
            child = self.crossover(parent1, parent2)
            mutated_child = self.mutate(child, rest_rate=rest_rate)
            new_population.append(mutated_child)

        self.population = new_population