import graphene
import vg_api.schema


class Query(vg_api.schema.Query, graphene.ObjectType):
    pass

class Mutation(vg_api.schema.Mutation, graphene.ObjectType):
    pass


schema = graphene.Schema(query=Query, mutation=Mutation)
