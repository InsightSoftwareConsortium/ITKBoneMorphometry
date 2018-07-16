/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkBoneMorphometryFeaturesFilter_hxx
#define itkBoneMorphometryFeaturesFilter_hxx

#include "itkBoneMorphometryFeaturesFilter.h"

#include "itkImageScanlineIterator.h"
#include "itkProgressReporter.h"
#include "itkNeighborhoodAlgorithm.h"

namespace itk
{
template< typename TInputImage, typename TMaskImage >
BoneMorphometryFeaturesFilter< TInputImage, TMaskImage >
::BoneMorphometryFeaturesFilter():
  m_Threshold(1),
  m_Pp(0),
  m_Pl(0),
  m_PlX(0),
  m_PlY(0),
  m_PlZ(0)
{
  this->SetNumberOfRequiredInputs( 1 );
}

template< typename TInputImage, typename TMaskImage >
void
BoneMorphometryFeaturesFilter< TInputImage, TMaskImage >
::AllocateOutputs()
{
  // Pass the input through as the output
  InputImagePointer image =
    const_cast< TInputImage * >( this->GetInput() );

  this->GraftOutput(image);

  // Nothing that needs to be allocated for the remaining outputs
}

template< typename TInputImage, typename TMaskImage >
void
BoneMorphometryFeaturesFilter< TInputImage, TMaskImage >
::BeforeThreadedGenerateData()
{
  ThreadIdType numberOfThreads = this->GetNumberOfThreads();

  m_Pp = 0;
  m_Pl = 0;
  m_PlX = 0;
  m_PlY = 0;
  m_PlZ = 0;

  // Initialize atomics
  m_NumVoxelsInsideMask.store(0);
  m_NumBoneVoxels.store(0);
  m_NumX.store(0);
  m_NumY.store(0);
  m_NumZ.store(0);
  m_NumXO.store(0);
  m_NumYO.store(0);
  m_NumZO.store(0);
}

template< typename TInputImage, typename TMaskImage >
void
BoneMorphometryFeaturesFilter< TInputImage, TMaskImage >
::AfterThreadedGenerateData()
{
  SizeValueType numVoxelsInsideMask = this->m_NumVoxelsInsideMask.load();
  SizeValueType numBoneVoxels = this->m_NumBoneVoxels.load();
  SizeValueType numX = this->m_NumX.load();
  SizeValueType numY = this->m_NumY.load();
  SizeValueType numZ = this->m_NumZ.load();
  SizeValueType numXO = this->m_NumXO.load();
  SizeValueType numYO = this->m_NumYO.load();
  SizeValueType numZO = this->m_NumZO.load();

  typename TInputImage::SpacingType inSpacing = this->GetInput()->GetSpacing();
  m_Pp = numBoneVoxels / static_cast<RealType>(numVoxelsInsideMask);
  m_PlX = ((numX+numXO)/2.0) / (numVoxelsInsideMask * inSpacing[0]) * 2;
  m_PlY = ((numY+numYO)/2.0) / (numVoxelsInsideMask * inSpacing[1]) * 2;
  m_PlZ = ((numZ+numZO)/2.0) / (numVoxelsInsideMask * inSpacing[2]) * 2;
  m_Pl =  (m_PlX + m_PlY + m_PlZ) / 3.0;
}

template< typename TInputImage, typename TMaskImage >
void
BoneMorphometryFeaturesFilter< TInputImage, TMaskImage >
::DynamicThreadedGenerateData(const RegionType & outputRegionForThread)
{
  NeighborhoodRadiusType radius;
  radius.Fill(1);
  NeighborhoodOffsetType offsetX = {{0,0,1}};
  NeighborhoodOffsetType offsetXO = {{0,0,-1}};
  NeighborhoodOffsetType offsetY = {{0,1,0}};
  NeighborhoodOffsetType offsetYO = {{0,-1,0}};
  NeighborhoodOffsetType offsetZ = {{1,0,0}};
  NeighborhoodOffsetType offsetZO = {{-1,0,0}};

  SizeValueType numVoxelsInsideMask = 0;
  SizeValueType numBoneVoxels = 0;
  SizeValueType numX = 0;
  SizeValueType numY = 0;
  SizeValueType numZ = 0;
  SizeValueType numXO = 0;
  SizeValueType numYO = 0;
  SizeValueType numZO = 0;

  MaskImagePointer maskPointer = TMaskImage::New();
  maskPointer = const_cast<TMaskImage*>(this->GetMaskImage());

  NeighborhoodAlgorithm::ImageBoundaryFacesCalculator< TInputImage > boundaryFacesCalculator;
  typename NeighborhoodAlgorithm::ImageBoundaryFacesCalculator< TInputImage >::FaceListType
  faceList = boundaryFacesCalculator( this->GetInput(), outputRegionForThread, radius );
  typename NeighborhoodAlgorithm::ImageBoundaryFacesCalculator< TInputImage >::FaceListType::iterator fit = faceList.begin();

  for (; fit != faceList.end(); ++fit )
    {
    NeighborhoodIteratorType inputNIt(radius, this->GetInput(), *fit );
    BoundaryConditionType  BoundaryCondition;
    inputNIt.SetBoundaryCondition(BoundaryCondition);
    inputNIt.GoToBegin();

    while( !inputNIt.IsAtEnd() )
      {
      if( maskPointer && maskPointer->GetPixel( inputNIt.GetIndex() ) == 0 )
        {
        ++inputNIt;
        continue;
        }

      ++numVoxelsInsideMask;

      if( inputNIt.GetCenterPixel() >= this->m_Threshold )
        {

        ++numBoneVoxels;

        if( inputNIt.GetPixel(offsetX) < this->m_Threshold )
          {
          ++numXO;
          }
        if( inputNIt.GetPixel(offsetXO) < this->m_Threshold )
          {
          ++numX;
          }
        if( inputNIt.GetPixel(offsetY) < this->m_Threshold )
          {
          ++numYO;
          }
        if( inputNIt.GetPixel(offsetYO) < this->m_Threshold )
          {
          ++numY;
          }
        if( inputNIt.GetPixel(offsetZ) < this->m_Threshold )
          {
          ++numZO;
          }
        if( inputNIt.GetPixel(offsetZO) < this->m_Threshold )
          {
          ++numZ;
          }
      }

      ++inputNIt;
      }
    }

  this->m_NumVoxelsInsideMask.fetch_add(numVoxelsInsideMask, std::memory_order_relaxed);
  this->m_NumBoneVoxels.fetch_add(numBoneVoxels, std::memory_order_relaxed);
  this->m_NumX.fetch_add(numX, std::memory_order_relaxed);
  this->m_NumY.fetch_add(numY, std::memory_order_relaxed);
  this->m_NumZ.fetch_add(numZ, std::memory_order_relaxed);
  this->m_NumXO.fetch_add(numXO, std::memory_order_relaxed);
  this->m_NumYO.fetch_add(numYO, std::memory_order_relaxed);
  this->m_NumZO.fetch_add(numZO, std::memory_order_relaxed);
}

template< typename TInputImage, typename TMaskImage >
void
BoneMorphometryFeaturesFilter< TInputImage, TMaskImage >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "m_Threshold: " << this->m_Threshold << std::endl;
  os << indent << "m_Pp: " << this->m_Pp << std::endl;
  os << indent << "m_Pl: " << this->m_Pl << std::endl;
  os << indent << "m_PlX: " << this->m_PlX << std::endl;
  os << indent << "m_PlY: " << this->m_PlY << std::endl;
  os << indent << "m_PlZ: " << this->m_PlZ << std::endl;
  os << indent << "m_NumVoxelsInsideMask: " << this->m_NumVoxelsInsideMask.load() << std::endl;
  os << indent << "m_NumBoneVoxels: " << this->m_NumBoneVoxels.load() << std::endl;
  os << indent << "m_NumX: " << this->m_NumX.load() << std::endl;
  os << indent << "m_NumY: " << this->m_NumY.load() << std::endl;
  os << indent << "m_NumZ: " << this->m_NumZ.load() << std::endl;
  os << indent << "m_NumXO: " << this->m_NumXO.load() << std::endl;
  os << indent << "m_NumYO: " << this->m_NumYO.load() << std::endl;
  os << indent << "m_NumZO: " << this->m_NumZO.load() << std::endl;

}
} // end namespace itk

#endif // itkBoneMorphometryFeaturesFilter_hxx
