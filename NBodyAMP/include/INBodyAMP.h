#pragma once

#include <vector>
#include <memory>

struct TaskData;

struct INBodyAMP
{
	virtual int tileSize() const = 0;
	virtual void integrate( TaskData& particleData, int numParticles, float deltaTime ) const = 0;
};