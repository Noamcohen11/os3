#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include "Barrier.h"
#include <atomic>
#include <pthread.h>
#include <cstdio>
#include <algorithm>
#include <semaphore.h>
#include <queue>
#include <iostream>

// TODO static cast
uint64_t STAGE_INC = (1ULL << 62);
uint64_t MASK = 0x000000007FFFFFFF; // 0x7FFFFFFF in hexadecimal

struct ThreadContext
{
	int threadID;
	std::atomic<uint64_t> *progress_counter;
	std::atomic<uint64_t> *map_counter;
	const InputVec *inputVec;
	const MapReduceClient *client;
	Barrier *barrier;
	sem_t *semaphore;
	int multiThreadLevel;
	IntermediateVec **interVec;
	pthread_mutex_t mutex_reduce;
	pthread_mutex_t mutex_emit;
	OutputVec outputVec;
};

struct JobHandleStruct
{
	pthread_t *threads;
	std::atomic<uint64_t> *progress_counter;
	sem_t *semaphore;
	int multiThreadLevel;
	IntermediateVec **interVec;
	pthread_mutex_t mutex_reduce;
	pthread_mutex_t mutex_emit;
};

// TODO understand counter.
void emit2(K2 *key, V2 *value, void *context)
{
	ThreadContext *tc = (ThreadContext *)context;
	IntermediatePair pair = std::make_pair(key, value);
	tc->interVec[tc->threadID]->push_back(pair);
	tc->progress_counter++;
}

void emit3(K3 *key, V3 *value, void *context)
{
	ThreadContext *tc = (ThreadContext *)context;
	pthread_mutex_lock(&tc->mutex_emit);
	OutputPair pair = std::make_pair(key, value);
	tc->outputVec.push_back(pair);
	pthread_mutex_unlock(&tc->mutex_emit);
}

K2 *findMaxKeyInLastPairs(IntermediateVec **interVec, int multiThreadLevel)
{
	K2 *maxKey = nullptr;

	for (int i = 0; i < multiThreadLevel; ++i)
	{
		// Ensure the vector is not empty before accessing its last element
		if (!interVec[i]->empty())
		{
			// Get a reference to the last pair in the current IntermediateVec
			IntermediatePair &lastPair = interVec[i]->back();
			K2 *currentKey = lastPair.first;

			// Compare currentKey with maxKey to find the maximum
			if (maxKey == nullptr || (*maxKey) < (*currentKey))
			{
				maxKey = currentKey;
			}
		}
	}

	return maxKey;
}

#include <bitset>

// Function to print binary representation of a 32-bit integer
void printBinary(uint32_t value, const char *label)
{
	std::cout << label << ": " << std::bitset<32>(value) << std::endl;
}

float calculateProgress(std::atomic<uint64_t> *counter)
{
	// MASK to isolate the lower 31 bits

	// Extract the first 31 bits and the next 31 bits
	uint32_t first31 = static_cast<uint32_t>((*counter >> 31) & MASK);
	uint32_t next31 = static_cast<uint32_t>(*counter & MASK);

	// Compare the first 31 bits with the next 31 bits
	// Print binary representation of the first 31 bits and the next 31 bits
	// printBinary(first31, "First 31 bits");
	// printBinary(next31, "Next 31 bits");
	return next31 / first31;
}

std::queue<IntermediateVec> __shuffle(ThreadContext *tc)
{
	std::queue<IntermediateVec> queue;
	while (calculateProgress(tc->progress_counter) != 1)
	{
		K2 *max_key = findMaxKeyInLastPairs(tc->interVec, tc->multiThreadLevel);
		for (int i = 0; i < tc->multiThreadLevel; i++)
		{
			IntermediateVec vec;
			// Ensure the vector is not empty before accessing its last element
			if (!tc->interVec[i]->empty())
			{
				// Get a const reference to the last pair in the current IntermediateVec
				const IntermediatePair &lastPair = tc->interVec[i]->back();
				K2 *currentKey = lastPair.first;

				// Compare currentKey with maxKey to find the maximum
				if (!(max_key < currentKey))
				{
					tc->interVec[i]->pop_back();
					vec.push_back(lastPair);
					tc->progress_counter++;
				}
			}
			queue.push(vec);
		}
	}
	return queue;
}

void reduce_func(ThreadContext *tc, std::queue<IntermediateVec> queue)
{
	while (!(queue.empty()))
	{
		pthread_mutex_lock(&tc->mutex_reduce);
		IntermediateVec vec = queue.front();
		queue.pop();
		pthread_mutex_unlock(&tc->mutex_reduce);

		tc->client->reduce(&vec, tc);
		tc->progress_counter += vec.size();
	}
}
// TODO add job precentage.
void *job_func(void *arg)
{
	ThreadContext *tc = (ThreadContext *)arg;

	// TODO: check if ++ is prefix or postfix.
	uint64_t old_value = (*(tc->map_counter))++;
	old_value = old_value & MASK;
	(void)old_value;
	while (old_value < (tc->inputVec)->size())
	{
		uint64_t old_value = (*(tc->map_counter))++;
		old_value = old_value & MASK;
		(void)old_value;
		// printf("premap thread %d \n", tc->threadID);
		tc->client->map((*(tc->inputVec))[old_value].first, (*(tc->inputVec))[old_value].second, tc);
	}
	// printf("postmap thread %d \n", tc->threadID);
	if (tc->interVec[tc->threadID]->empty())
	{
		printf("sort size %d \n", tc->interVec[tc->threadID]->size());
		// std::sort(tc->interVec[tc->threadID]->begin(), tc->interVec[tc->threadID]->end());
	}
	tc->barrier->barrier();
	std::queue<IntermediateVec> queue;
	if (tc->threadID == 0)
	{
		int shuffle_keys = 0;
		for (int i = 0; i < tc->multiThreadLevel; i++)
		{
			shuffle_keys += tc->interVec[tc->threadID]->size();
		}
		tc->progress_counter = 0;
		tc->progress_counter += (2ULL << 62);
		tc->progress_counter += (shuffle_keys << 31);
		queue = __shuffle(tc);
		uint64_t new_count = static_cast<uint64_t>((*tc->progress_counter >> 31) & MASK);
		tc->progress_counter = 0;
		tc->progress_counter += new_count;
		tc->progress_counter += (3ULL << 62);
		for (int i = 0; i < tc->multiThreadLevel; ++i)
		{
			sem_post(tc->semaphore); // Post to unblock one waiting thread
		}
	}
	else
	{
		sem_wait(tc->semaphore);
	}
	reduce_func(tc, queue);
	return nullptr;
}

JobHandle startMapReduceJob(const MapReduceClient &client,
							const InputVec &inputVec, OutputVec &outputVec,
							int multiThreadLevel)
{
	pthread_t threads[multiThreadLevel];
	ThreadContext contexts[multiThreadLevel];
	IntermediateVec **interVec = new IntermediateVec *[multiThreadLevel];
	Barrier barrier(multiThreadLevel);
	std::atomic<uint64_t> progress_counter(0);
	std::atomic<uint64_t> map_counter(0);
	progress_counter += STAGE_INC;
	progress_counter += ((&inputVec)->size()) << 31;
	sem_t semaphore;
	pthread_mutex_t mutex_reduce;
	pthread_mutex_t mutex_emit;
	sem_init(&semaphore, 0, 0);
	for (int i = 0; i < multiThreadLevel; ++i)
	{
		contexts[i] = {
			i,
			&progress_counter,
			&map_counter,
			&inputVec,
			&client,
			&barrier,
			&semaphore,
			multiThreadLevel,
			interVec,
			mutex_reduce,
			mutex_emit,
			outputVec};
	}
	for (int i = 0; i < multiThreadLevel; ++i)
	{
		pthread_create(threads + i, NULL, job_func, contexts + i);
	}
	JobHandleStruct *job = new JobHandleStruct{threads, &progress_counter, &semaphore, multiThreadLevel, interVec, mutex_reduce, mutex_emit};
	return job;
}

void waitForJob(JobHandle job)
{
	JobHandleStruct *job_st = (JobHandleStruct *)job;
	for (int i = 0; i < job_st->multiThreadLevel; i++)
	{
		if (pthread_join(job_st->threads[i], nullptr))
		{
			std::cout << "couldn't join the thread" << std::endl;
			exit(1);
		}
	}
}
void getJobState(JobHandle job, JobState *state)
{
	JobHandleStruct *job_st = (JobHandleStruct *)job;
	state->percentage = calculateProgress(job_st->progress_counter) * 100;
	state->stage = static_cast<stage_t>(*job_st->progress_counter >> 62);
}
// TODO sem_destroy(&semaphore);
// TODO mutex_destroy(&mutex);
void closeJobHandle(JobHandle job)
{
	waitForJob(job);
	JobHandleStruct *job_st = (JobHandleStruct *)job;
	pthread_mutex_destroy(&job_st->mutex_emit);
	pthread_mutex_destroy(&job_st->mutex_reduce);
	sem_destroy(job_st->semaphore);

	for (int i = 0; i < job_st->multiThreadLevel; ++i)
	{
		delete job_st->interVec[i];
	}
	delete[] job_st->interVec;
	delete job_st;
}
