#include "MapReduceFramework.h"
#include "Barrier.h"
#include "MapReduceClient.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <pthread.h>
#include <queue>
#include <semaphore.h>
#include <unistd.h>

// Constants
uint64_t STAGE_INC = (1ULL << 62);
uint64_t MASK = 0x000000007FFFFFFF; // 0x7FFFFFFF in hexadecimal

// Thread context structure
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
  pthread_mutex_t *mutex_reduce;
  pthread_mutex_t *mutex_emit;
  OutputVec *outputVec;
};

// Job handle structure
struct JobHandleStruct
{
  pthread_t *threads;
  std::atomic<uint64_t> *progress_counter;
  sem_t *semaphore;
  int multiThreadLevel;
  IntermediateVec **interVec;
  pthread_mutex_t *mutex_reduce;
  pthread_mutex_t *mutex_emit;
};

// Emit2 function
void emit2(K2 *key, V2 *value, void *context)
{
  ThreadContext *tc = (ThreadContext *)context;
  IntermediatePair pair = std::make_pair(key, value);
  tc->interVec[tc->threadID]->push_back(pair);
}

// Emit3 function
void emit3(K3 *key, V3 *value, void *context)
{
  ThreadContext *tc = (ThreadContext *)context;
  pthread_mutex_lock(tc->mutex_emit);
  OutputPair pair = std::make_pair(key, value);
  tc->outputVec->push_back(pair);
  pthread_mutex_unlock(tc->mutex_emit);
}

// Helper function to find the maximum key in the last pairs of intermediate vectors
K2 *findMaxKeyInLastPairs(IntermediateVec **interVec, int multiThreadLevel)
{
  K2 *maxKey = nullptr;
  for (int i = 0; i < multiThreadLevel; ++i)
  {
    if (!interVec[i]->empty())
    {
      IntermediatePair &lastPair = interVec[i]->back();
      K2 *currentKey = lastPair.first;
      if (maxKey == nullptr || (*maxKey) < (*currentKey))
      {
        maxKey = currentKey;
      }
    }
  }
  return maxKey;
}

void printFirstAndNext(std::atomic<uint64_t> *counter)
{
  const uint64_t MASK = 0x7FFFFFFF; // MASK to isolate the lower 31 bits

  // Extract the first 31 bits and the next 31 bits
  uint32_t stage = static_cast<uint32_t>((*counter >> 62));
  uint32_t first31 = static_cast<uint32_t>((*counter >> 31) & MASK);
  uint32_t next31 = static_cast<uint32_t>(*counter & MASK);

  // Print the first and next 31 bits
  std::cout << "The stage " << stage << std::endl;
  std::cout << "First 31 bits: " << first31 << std::endl;
  std::cout << "Next 31 bits: " << next31 << std::endl;
}

float calculateProgress(std::atomic<uint64_t> *counter)
{
  printFirstAndNext(counter);
  uint32_t first31 = static_cast<uint32_t>((*counter >> 31) & MASK);
  uint32_t next31 = static_cast<uint32_t>(*counter & MASK);
  return first31 != 0 ? static_cast<float>(next31) / first31 : 0.0f;
}

// Shuffle function
std::queue<IntermediateVec> __shuffle(ThreadContext *tc)
{
  std::queue<IntermediateVec> queue;
  while (calculateProgress(tc->progress_counter) != 1)
  {
    K2 *max_key = findMaxKeyInLastPairs(tc->interVec, tc->multiThreadLevel);
    for (int i = 0; i < tc->multiThreadLevel; i++)
    {
      IntermediateVec vec;
      if (!tc->interVec[i]->empty())
      {
        const IntermediatePair &lastPair = tc->interVec[i]->back();
        K2 *currentKey = lastPair.first;
        if (!(max_key < currentKey))
        {
          tc->interVec[i]->pop_back();
          vec.push_back(lastPair);
          (*(tc->progress_counter))++;
        }
      }
      queue.push(vec);
    }
  }
  return queue;
}

// Reduce function
void reduce_func(ThreadContext *tc, std::queue<IntermediateVec> queue)
{
  while (!queue.empty())
  {
    pthread_mutex_lock(tc->mutex_reduce);
    IntermediateVec vec = queue.front();
    queue.pop();
    pthread_mutex_unlock(tc->mutex_reduce);
    tc->client->reduce(&vec, tc);
    (*(tc->progress_counter)) += vec.size();
  }
}

// Job function
void *job_func(void *arg)
{
  ThreadContext *tc = (ThreadContext *)arg;
  uint64_t old_value = (*(tc->map_counter))++;
  old_value = old_value & MASK;
  while (old_value < tc->inputVec->size())
  {
    printf("premap thread %d\n", tc->threadID);
    tc->client->map((*(tc->inputVec))[old_value].first,
                    (*(tc->inputVec))[old_value].second, tc);
    old_value = (*(tc->map_counter))++;
    (*(tc->progress_counter))++;
    old_value = old_value & MASK;
  }
  printf("thread %d done mapping\n", tc->threadID);
  if (!tc->interVec[tc->threadID]->empty())
  {
    std::sort(tc->interVec[tc->threadID]->begin(),
              tc->interVec[tc->threadID]->end());
  }
  tc->barrier->barrier();
  std::queue<IntermediateVec> queue;
  if (tc->threadID == 0)
  {
    int shuffle_keys = 0;
    for (int i = 0; i < tc->multiThreadLevel; i++)
    {
      shuffle_keys += tc->interVec[i]->size();
    }
    *(tc->progress_counter) = 0;
    *(tc->progress_counter) += (2ULL << 62);
    *(tc->progress_counter) += (shuffle_keys << 31);
    queue = __shuffle(tc);
    uint64_t new_count =
        static_cast<uint64_t>((*(tc->progress_counter) >> 31) & MASK);
    *(tc->progress_counter) = 0;
    *(tc->progress_counter) += new_count;
    *(tc->progress_counter) += (3ULL << 62);
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

// Start MapReduce job
JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec, OutputVec &outputVec,
                            int multiThreadLevel)
{
  pthread_t *threads = new pthread_t[multiThreadLevel];
  ThreadContext *contexts = new ThreadContext[multiThreadLevel];
  IntermediateVec **interVec = new IntermediateVec *[multiThreadLevel];
  Barrier *barrier = new Barrier(multiThreadLevel);
  std::atomic<uint64_t> *progress_counter = new std::atomic<uint64_t>(0);
  std::atomic<uint64_t> *map_counter = new std::atomic<uint64_t>(0);
  sem_t *semaphore = new sem_t;
  pthread_mutex_t *mutex_reduce = new pthread_mutex_t;
  pthread_mutex_t *mutex_emit = new pthread_mutex_t;
  sem_init(semaphore, 0, 0);
  pthread_mutex_init(mutex_reduce, nullptr);
  pthread_mutex_init(mutex_emit, nullptr);
  *(progress_counter) += STAGE_INC;
  *(progress_counter) += inputVec.size() << 31;
  for (int i = 0; i < multiThreadLevel; ++i)
  {
    interVec[i] = new IntermediateVec();
    contexts[i] = {i, progress_counter, map_counter, &inputVec,
                   &client, barrier, semaphore, multiThreadLevel,
                   interVec, mutex_reduce, mutex_emit, &outputVec};
  }
  for (int i = 0; i < multiThreadLevel; ++i)
  {
    printf("Creating thread %d\n", i);
    pthread_create(threads + i, nullptr, job_func, contexts + i);
  }
  JobHandleStruct *job = new JobHandleStruct{
      threads, progress_counter, semaphore, multiThreadLevel,
      interVec, mutex_reduce, mutex_emit};
  return job;
}

// Wait for job to finish
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

// Get job state
void getJobState(JobHandle job, JobState *state)
{
  JobHandleStruct *job_st = (JobHandleStruct *)job;
  state->percentage = calculateProgress(job_st->progress_counter) * 100;
  state->stage = static_cast<stage_t>(*(job_st->progress_counter) >> 62);
}

// Close job handle
void closeJobHandle(JobHandle job)
{
  waitForJob(job);
  JobHandleStruct *job_st = (JobHandleStruct *)job;
  pthread_mutex_destroy(job_st->mutex_emit);
  pthread_mutex_destroy(job_st->mutex_reduce);
  sem_destroy(job_st->semaphore);
  for (int i = 0; i < job_st->multiThreadLevel; ++i)
  {
    delete job_st->interVec[i];
  }
  delete[] job_st->interVec;
  delete job_st->threads;
  delete job_st->progress_counter;
  delete job_st->semaphore;
  delete job_st->mutex_reduce;
  delete job_st->mutex_emit;
  delete job_st;
}
