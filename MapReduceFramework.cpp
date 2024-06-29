#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include "Barrier.h"
#include <atomic>
#include <pthread.h>
#include <cstdio>
#include <algorithm>

struct ThreadContext
{
	int threadID;
	std::atomic<uint32_t> *atomic_counter;
	const InputVec *inputVec;
	const MapReduceClient *client;
	IntermediateVec *interVec;
	Barrier *barrier;
};

void emit2(K2 *key, V2 *value, void *context)
{
	ThreadContext *tc = (ThreadContext *)context;
	IntermediatePair pair = std::make_pair(key, value);
	tc->interVec->push_back(pair);
}

void emit3(K3 *key, V3 *value, void *context);

// TODO add job precentage.
void *job_func(void *arg)
{
	ThreadContext *tc = (ThreadContext *)arg;

	// TODO: check if ++ is prefix or postfix.
	int old_value = ++(*(tc->atomic_counter));
	(void)old_value;
	while (old_value < (tc->inputVec)->size())
	{
		int old_value = ++(*(tc->atomic_counter));
		(void)old_value;

		tc->client->map((*(tc->inputVec))[old_value].first, (*(tc->inputVec))[old_value].second, tc);
	}
	std::sort(tc->interVec->begin(), tc->interVec->end());
	tc->barrier->barrier();
	return arg;
}

JobHandle startMapReduceJob(const MapReduceClient &client,
							const InputVec &inputVec, OutputVec &outputVec,
							int multiThreadLevel)
{
	pthread_t threads[multiThreadLevel];
	ThreadContext contexts[multiThreadLevel];
	IntermediateVec interVec[multiThreadLevel];
	Barrier barrier(multiThreadLevel);
	std::atomic<uint32_t> atomic_counter(0);

	for (int i = 0; i < multiThreadLevel; ++i)
	{
		contexts[i] = {i, &atomic_counter, &inputVec, &client, interVec + i, &barrier};
	}

	for (int i = 0; i < multiThreadLevel; ++i)
	{
		pthread_create(threads + i, NULL, job_func, contexts + i);
	}
	JobHandle job;
	return job;
}

void waitForJob(JobHandle job);
// {
// 	// pthread_join(threads[i], NULL);
// }
void getJobState(JobHandle job, JobState *state);
void closeJobHandle(JobHandle job);
