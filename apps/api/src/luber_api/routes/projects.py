"""Project endpoints: a lightweight workspace grouping for generations.

A project is a folder. It has a name and it holds generations. There is
no collaboration, no sharing and no permission model here — those are
later phases, and guessing at their shape now would be inventing
requirements.

Deleting a project never deletes the music inside it: the foreign key is
``ON DELETE SET NULL`` and the generations return to being unfiled.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from luber_api.dependencies import get_repository
from luber_api.routes.generations import serialize_generation
from luber_api.schemas import (
    AssignProjectRequest,
    GenerationListResponse,
    GenerationResponse,
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectResponse,
)
from luber_database import GenerationRepository

router = APIRouter(prefix="/v1/projects", tags=["projects"])


async def _to_response(repository: GenerationRepository, project: object) -> ProjectResponse:
    return ProjectResponse(
        id=project.id,  # type: ignore[attr-defined]
        name=project.name,  # type: ignore[attr-defined]
        generation_count=await repository.count_project_generations(
            project.id  # type: ignore[attr-defined]
        ),
        created_at=project.created_at,  # type: ignore[attr-defined]
        updated_at=project.updated_at,  # type: ignore[attr-defined]
    )


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ProjectResponse)
async def create_project(
    payload: ProjectCreateRequest,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> ProjectResponse:
    project = await repository.create_project(name=payload.name)
    return await _to_response(repository, project)


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> ProjectListResponse:
    """Every project with its song count, resolved in a single query."""
    return ProjectListResponse(
        items=[
            ProjectResponse(
                id=project.id,
                name=project.name,
                generation_count=count,
                created_at=project.created_at,
                updated_at=project.updated_at,
            )
            for project, count in await repository.list_projects_with_counts()
        ]
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: uuid.UUID,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> ProjectResponse:
    project = await repository.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="project not found")
    return await _to_response(repository, project)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def rename_project(
    project_id: uuid.UUID,
    payload: ProjectCreateRequest,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> ProjectResponse:
    try:
        project = await repository.rename_project(project_id, name=payload.name)
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="project not found"
        ) from None
    return await _to_response(repository, project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: uuid.UUID,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> None:
    """Delete the folder, keep the music."""
    try:
        await repository.delete_project(project_id)
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="project not found"
        ) from None


@router.get("/{project_id}/generations", response_model=GenerationListResponse)
async def list_project_generations(
    project_id: uuid.UUID,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> GenerationListResponse:
    if await repository.get_project(project_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="project not found")
    generations = await repository.list_generations_for_project(project_id)
    return GenerationListResponse(
        items=[serialize_generation(g) for g in generations],
        total=len(generations),
        limit=len(generations),
        offset=0,
    )


assign_router = APIRouter(prefix="/v1/generations", tags=["projects"])


@assign_router.put("/{generation_id}/project", response_model=GenerationResponse)
async def assign_generation_to_project(
    generation_id: uuid.UUID,
    payload: AssignProjectRequest,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> GenerationResponse:
    """File a generation under a project, or unfile it with ``null``.

    Returns the updated generation. Phase 11 wrapped this single row in a
    list response, which made every caller unwrap a collection that could
    only ever hold one item.
    """
    if await repository.get_generation(generation_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="generation not found")
    if payload.project_id is not None and await repository.get_project(payload.project_id) is None:
        raise HTTPException(status_code=422, detail="project not found")

    await repository.set_generation_project(generation_id, payload.project_id)
    generation = await repository.get_generation(generation_id)
    assert generation is not None
    return serialize_generation(generation)
