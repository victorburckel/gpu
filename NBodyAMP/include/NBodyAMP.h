#pragma once

#include "INBodyAMP.h"

struct NBodyAMP : public INBodyAMP
{
	virtual int tileSize() const override;
	virtual void integrate( TaskData& particleData, int numParticles, float deltaTime ) const override;
};
