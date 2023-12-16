import graphene
from django.conf import settings
from vg_api.util import generate_team


class PokemonType(graphene.ObjectType):
    name = graphene.String()
    hold_item = graphene.String()
    moveset = graphene.List(graphene.String)


class TeamType(graphene.ObjectType):
    members = graphene.List(PokemonType)


class Query:
    version = graphene.String()
    def resolve_version(self, info, **kwargs):
        return settings.VERSION


class GenerateVGCTeam(graphene.relay.ClientIDMutation):
    team = graphene.Field(TeamType)

    class Input:
        seed = graphene.List(
            graphene.String,
            required=True
        )
        date_reference = graphene.Date(
            required=True
        )

    def mutate_and_get_payload(self, info, **kwargs):
        seed = [i.title() for i in kwargs['seed']]

        if not seed:
            raise Exception('A pokemon name is required as seed.')

        year, month, _ = str(kwargs['date_reference']).split('-')
        url = f'https://www.smogon.com/stats/{year}-{month}/moveset/gen9vgc{year}regulatione-1760.txt'
        team = generate_team(url, seed)

        return GenerateVGCTeam(team)


class Mutation:
    generate_vgc_team = GenerateVGCTeam.Field()
