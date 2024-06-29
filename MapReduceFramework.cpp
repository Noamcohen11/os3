#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include "Barrier.h"
#include <atomic>
#include <pthread.h>
#include <cstdio>
#include <algorithm>
#include <semaphore.h>
#include <queue>

uint64_t STAGE_INC = (1 << 62);
uint64_t TOTAL_INC = (1 << 31);
uint64_t RESET_COUNT = (3 << 62);

struct ThreadContext
{
	int threadID;
	std::atomic<uint64_t> *progress_counter;
	const InputVec *inputVec;
	const MapReduceClient *client;
	Barrier *barrier;
	sem_t *semaphore;
	int multiThreadLevel;
	IntermediateVec **interVec;
};

// TODO understand counter.
void emit2(K2 *key, V2 *value, void *context)
{
	ThreadContext *tc = (ThreadContext *)context;
	IntermediatePair pair = std::make_pair(key, value);
	tc->interVec[tc->threadID]->push_back(pair);
}

void emit3(K3 *key, V3 *value, void *context);

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
bool compareProgress(std::atomic<uint64_t> *counter)
{
	// Mask to isolate the lower 31 bits
	uint64_t mask = 0x000000007FFFFFFF; // 0x7FFFFFFF in hexadecimal

	// Extract the first 31 bits and the next 31 bits
	uint32_t first31 = static_cast<uint32_t>((*counter >> 31) & mask);
	uint32_t next31 = static_cast<uint32_t>(*counter & mask);

	// Compare the first 31 bits with the next 31 bits
	return first31 == next31;
}

std::queue<IntermediateVec> __shuffle(ThreadContext *tc)
{
	std::queue<IntermediateVec> queue;
	while (!compareProgress(tc->progress_counter))
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

// TODO add job precentage.
void *job_func(void *arg)
{
	ThreadContext *tc = (ThreadContext *)arg;

	// TODO: check if ++ is prefix or postfix.
	int old_value = ++(*(tc->progress_counter));
	(void)old_value;
	while (old_value < (tc->inputVec)->size())
	{
		int old_value = ++(*(tc->progress_counter));
		(void)old_value;

		tc->client->map((*(tc->inputVec))[old_value].first, (*(tc->inputVec))[old_value].second, tc);
	}
	std::sort(tc->interVec[tc->threadID]->begin(), tc->interVec[tc->threadID]->end());
	tc->barrier->barrier();
	if (tc->threadID == 0)
	{
		int shuffle_keys = 0;
		for (int i = 0; i < tc->multiThreadLevel; i++)
		{
			shuffle_keys += tc->interVec[tc->threadID]->size();
		}
		tc->progress_counter = 0;
		tc->progress_counter += (2 << 62);
		tc->progress_counter += (shuffle_keys << 31);
		std::queue<IntermediateVec> queue = __shuffle(tc);
		tc->progress_counter = 0;
		tc->progress_counter += (3 << 62);
		for (int i = 0; i < tc->multiThreadLevel; ++i)
		{
			sem_post(tc->semaphore); // Post to unblock one waiting thread
		}
	}
	else
	{
		sem_wait(tc->semaphore);
	}
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
	progress_counter += STAGE_INC;
	progress_counter += ((&inputVec)->size()) << 31;
	sem_t semaphore;
	sem_init(&semaphore, 0, 0);

	for (int i = 0; i < multiThreadLevel; ++i)
	{
		contexts[i] = {i, &progress_counter, &inputVec, &client, &barrier, &semaphore, multiThreadLevel, interVec};
	}

	for (int i = 0; i < multiThreadLevel; ++i)
	{
		pthread_create(threads + i, NULL, job_func, contexts + i);
	}
}

void waitForJob(JobHandle job);
// {
// 	// pthread_join(threads[i], NULL);
// }
void getJobState(JobHandle job, JobState *state);
// TODO sem_destroy(&semaphore);
void closeJobHandle(JobHandle job);
